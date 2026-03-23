"""Demo: audio-driven empathetic talking-head generation.

Given a **listener** identity image and audio, plus a **speaker** video,
this script generates a listener video whose expressions empathetically
reflect the speaker.  Because the model includes a stochastic CVAE, each
run produces a *different* empathetic response.

Usage::

    python demo_empathy.py \\
        --listener_source_path test_data/listener_identity.jpg \\
        --listener_audio_path  test_data/listener_audio.wav \\
        --speaker_video_path   test_data/speaker_video.mp4 \\
        --pose_driving_path    test_data/pose_source1.mp4 \\
        --model_path           ckpts/empathy.pt \\
        --audio2lip_model_path ckpts/Audio2Lip.pt \\
        --save_path            res/demo_empathy.mp4
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from networks.empathy_generator import EmpathyGenerator
from networks.audio_encoder import Audio2Lip
from networks.utils import check_package_installed

import audio


# ------------------------------------------------------------------
# Preprocessing helpers (shared with other demo scripts)
# ------------------------------------------------------------------

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x H x W
    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return (img - 0.5) * 2.0  # [-1, 1]


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    transform = transforms.Compose([transforms.Resize((256, 256))])
    resized = torch.stack(
        [transform(frame) for frame in vid_norm[0]], dim=0
    ).unsqueeze(0)
    return resized, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type(
        'torch.ByteTensor'
    )
    torchvision.io.write_video(save_path, vid[0], fps=fps)


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps
    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    return audio_length, num_frames


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant',
                     constant_values=0)
    return wav


def get_mel(audio_path):
    wav = audio.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()
    indiv_mels = []
    fps = 25
    syncnet_mel_step_size = 16
    for i in range(num_frames):
        start_frame_num = i - 2
        start_idx = int(80.0 * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0)
    source_audio_feature = indiv_mels
    mel_input = source_audio_feature
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)
    return audiox, bs, T


def conv_feat(features, k_size, sigma=1.0):
    c = features.shape[1]
    pad = k_size // 2
    k = np.zeros(k_size).astype(np.float64)
    for x in range(-pad, k_size - pad):
        k[x + pad] = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1)
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)
    return features


# ------------------------------------------------------------------
# Demo class
# ------------------------------------------------------------------

class EmpathyDemo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        print('==> loading models')
        # Audio-to-lip
        self.audio2lip = Audio2Lip()
        a2l_weight = torch.load(
            args.audio2lip_model_path, map_location='cpu'
        )['audio2lip']
        self.audio2lip.load_state_dict(a2l_weight)
        self.audio2lip.eval()

        # Empathy generator
        self.gen = EmpathyGenerator(
            size=args.size,
            style_dim=args.latent_dim_style,
            lip_dim=args.latent_dim_lip,
            pose_dim=args.latent_dim_pose,
            exp_dim=args.latent_dim_exp,
            channel_multiplier=args.channel_multiplier,
            emotion_dim=args.emotion_dim,
            cvae_latent_dim=args.cvae_latent_dim,
        )
        weight = torch.load(args.model_path, map_location='cpu')
        # Support loading from either a full empathy checkpoint or a
        # backbone-only checkpoint
        if 'gen' in weight:
            self.gen.load_state_dict(weight['gen'], strict=False)
        self.gen.eval()

        if torch.cuda.is_available():
            self.audio2lip = self.audio2lip.cuda()
            self.gen = self.gen.cuda()

        # Data
        print('==> loading data')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.listener_source = img_preprocessing(
            args.listener_source_path, args.size
        ).to(device)
        self.listener_audio, self.bs, self.T = get_mel(args.listener_audio_path)
        self.listener_audio = self.listener_audio.to(device)
        self.audio_path = args.listener_audio_path

        self.speaker_vid, self.fps = vid_preprocessing(args.speaker_video_path)
        self.speaker_vid = self.speaker_vid.to(device)

        self.pose_vid, _ = vid_preprocessing(args.pose_driving_path)
        self.pose_vid = self.pose_vid.to(device)

        self.save_path = args.save_path

    def run(self):
        print('==> running empathetic generation')
        with torch.no_grad():
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            # Listener lip coefficients from audio
            lip_coeffs = self.audio2lip(self.listener_audio, self.bs, self.T)[0]
            lip_coeffs = conv_feat(lip_coeffs, k_size=3, sigma=1)

            num_frames = lip_coeffs.size(0)
            spk_len = self.speaker_vid.shape[1]
            pose_len = self.pose_vid.shape[1]

            vid_target_recon = []
            for i in tqdm(range(num_frames)):
                lip_i = lip_coeffs[i:i + 1]
                pose_i = self.pose_vid[:, min(i, pose_len - 1)]
                spk_i = self.speaker_vid[:, min(i, spk_len - 1)]

                img_recon = self.gen.test_empathy_A(
                    listener_source=self.listener_source,
                    listener_lip_coeffs=lip_i,
                    listener_pose_frame=pose_i,
                    speaker_frame=spk_i,
                    speaker_mel=None,
                )
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)

            temp_path = self.save_path.replace('.mp4', '_temp.mp4')
            save_video(vid_target_recon, temp_path, self.fps)
            cmd = 'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (
                temp_path, self.audio_path, self.save_path
            )
            os.system(cmd)
            os.remove(temp_path)

            # Optional face super-resolution
            if self.args.face_sr and check_package_installed('gfpgan'):
                from face_sr.face_enhancer import enhancer_list
                from moviepy.editor import VideoFileClip, AudioFileClip
                import imageio

                temp_512 = self.save_path.replace('.mp4', '_512.mp4')
                imageio.mimsave(
                    temp_512 + '.tmp.mp4',
                    enhancer_list(self.save_path, method='gfpgan',
                                  bg_upsampler=None),
                    fps=float(25), codec='libx264',
                )
                video_clip = VideoFileClip(temp_512 + '.tmp.mp4')
                audio_clip = AudioFileClip(self.save_path)
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(temp_512, codec='libx264',
                                           audio_codec='aac')
                os.remove(temp_512 + '.tmp.mp4')

        print(f'==> saved to {self.save_path}')


# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Empathetic talking-head generation demo'
    )
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--latent_dim_style', type=int, default=512)
    parser.add_argument('--latent_dim_lip', type=int, default=20)
    parser.add_argument('--latent_dim_pose', type=int, default=6)
    parser.add_argument('--latent_dim_exp', type=int, default=10)
    parser.add_argument('--emotion_dim', type=int, default=128)
    parser.add_argument('--cvae_latent_dim', type=int, default=32)
    # Input paths
    parser.add_argument('--listener_source_path', type=str,
                        default='test_data/identity_source.jpg')
    parser.add_argument('--listener_audio_path', type=str,
                        default='test_data/mouth_source.wav')
    parser.add_argument('--speaker_video_path', type=str,
                        default='test_data/expression_source.mp4')
    parser.add_argument('--pose_driving_path', type=str,
                        default='test_data/pose_source1.mp4')
    # Model paths
    parser.add_argument('--model_path', type=str,
                        default='ckpts/empathy.pt')
    parser.add_argument('--audio2lip_model_path', type=str,
                        default='ckpts/Audio2Lip.pt')
    # Output
    parser.add_argument('--save_path', type=str,
                        default='res/demo_empathy.mp4')
    parser.add_argument('--face_sr', action='store_true')

    args = parser.parse_args()
    demo = EmpathyDemo(args)
    demo.run()
