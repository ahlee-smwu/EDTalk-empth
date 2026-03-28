"""
Extract Predefined Expression Weights from the Trained EDTalk Model.

This script extracts expression coefficient vectors (10-dim) for each emotion category
by running inference on emotion-labeled videos from the MEAD dataset using the trained
EDTalk model. The extracted weights are saved as .npy files in ckpts/predefined_exp_weights/.

How it works:
  1. Load the trained EDTalk generator model (ckpts/EDTalk.pt).
  2. For each emotion category (angry, contempt, disgusted, fear, happy, sad, surprised, neutral),
     collect all video frames labeled with that emotion from the MEAD dataset.
  3. For each frame, pass it through the encoder and the expression FC network (exp_fc)
     to extract a 10-dimensional expression coefficient vector (alpha_D_exp).
  4. Average all coefficient vectors for each emotion to get a single representative
     10-dim weight vector per emotion.
  5. Save each emotion's weight vector as a .npy file (e.g., angry.npy).

These saved .npy files are then used at inference time by demo_EDTalk_A_using_predefined_exp_weights.py
via the --exp_type argument (e.g., --exp_type angry loads ckpts/predefined_exp_weights/angry.npy).

Usage:
  python extract_predefined_exp_weights.py \
    --model_path ckpts/EDTalk.pt \
    --data_root /path/to/MEAD_front/crop_frame \
    --save_dir ckpts/predefined_exp_weights

Requirements:
  - The trained EDTalk model checkpoint (ckpts/EDTalk.pt)
  - Cropped and preprocessed MEAD dataset frames organized as:
      {data_root}/{person_id}/{emotion}/level_{x}/{clip_id}/{frame}.png

Architecture context (from networks/generator.py):
  - Generator.enc: Encoder that maps a face image to a 512-dim style vector (wa)
  - Generator.fc: Shared FC layers that project the style vector
  - Generator.exp_fc: Expression-specific FC layers that output 10-dim expression coefficients
  - Generator.direction_exp: Direction_exp module with learnable orthogonal basis (512 x 10)
    that projects the 10 expression coefficients into a 512-dim latent space via QR decomposition
"""

import os
import argparse
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from networks.generator import Generator


EMOTION_CATEGORIES = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'neutral']


def img_preprocessing(img_path, size=256):
    """Load and normalize an image to [-1, 1] range."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256
    img = img / 255.0
    img = torch.from_numpy(img).unsqueeze(0).float()
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]
    return imgs_norm


def collect_frames_by_emotion(data_root):
    """
    Collect frame paths grouped by emotion from MEAD dataset.

    Expected directory structure:
      {data_root}/{person_id}/{emotion}/level_{x}/{clip_id}/{frame}.png

    Returns:
      dict mapping emotion name -> list of frame file paths
    """
    emotion_frames = {emotion: [] for emotion in EMOTION_CATEGORIES}

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    for person_id in sorted(os.listdir(data_root)):
        person_dir = os.path.join(data_root, person_id)
        if not os.path.isdir(person_dir):
            continue
        for emotion in EMOTION_CATEGORIES:
            emotion_dir = os.path.join(person_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            frames = sorted(glob.glob(os.path.join(emotion_dir, '**', '*.png'), recursive=True))
            emotion_frames[emotion].extend(frames)

    return emotion_frames


def extract_exp_weights(gen, frames, device, max_samples=None):
    """
    Extract expression coefficients from frames using the trained generator.

    For each frame:
      1. Encode the image with gen.enc to get style vector wa_t
      2. Pass wa_t through gen.fc (shared FC) to get shared features
      3. Pass shared features through gen.exp_fc to get 10-dim expression coefficients

    Args:
        gen: Trained Generator model
        frames: List of frame file paths
        device: torch device
        max_samples: Optional limit on number of frames to process

    Returns:
        Mean expression coefficient vector of shape (1, 10)
    """
    if max_samples is not None:
        frames = frames[:max_samples]

    all_exp_coeffs = []

    for frame_path in tqdm(frames, desc="Extracting"):
        img = img_preprocessing(frame_path).to(device)
        with torch.no_grad():
            # Encode the image: enc returns (wa_source, wa_target, feats_source, feats_target)
            # When both inputs are the same, wa_t captures the style of the input image
            _, wa_t, _, _ = gen.enc(img, img)

            # Extract expression coefficients through FC layers
            shared_fc = gen.fc(wa_t)
            alpha_D_exp = gen.exp_fc(shared_fc)  # shape: (1, 10)

            all_exp_coeffs.append(alpha_D_exp.cpu().numpy())

    if len(all_exp_coeffs) == 0:
        return None

    # Average all expression coefficients for this emotion
    all_exp_coeffs = np.concatenate(all_exp_coeffs, axis=0)  # (N, 10)
    mean_exp = np.mean(all_exp_coeffs, axis=0, keepdims=True)  # (1, 10)

    return mean_exp.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Extract predefined expression weights from the trained EDTalk model using MEAD data.'
    )
    parser.add_argument('--model_path', type=str, default='ckpts/EDTalk.pt',
                        help='Path to the trained EDTalk model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of preprocessed MEAD dataset frames')
    parser.add_argument('--save_dir', type=str, default='ckpts/predefined_exp_weights',
                        help='Directory to save extracted expression weight .npy files')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size for preprocessing')
    parser.add_argument('--latent_dim_style', type=int, default=512)
    parser.add_argument('--latent_dim_lip', type=int, default=20)
    parser.add_argument('--latent_dim_pose', type=int, default=6)
    parser.add_argument('--latent_dim_exp', type=int, default=10)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--max_samples_per_emotion', type=int, default=None,
                        help='Maximum number of frames to process per emotion (for faster extraction)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Load the trained generator
    print('==> Loading trained EDTalk generator model')
    gen = Generator(
        args.size, args.latent_dim_style,
        args.latent_dim_lip, args.latent_dim_pose,
        args.latent_dim_exp, args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    gen.load_state_dict(checkpoint['gen'])
    gen.eval()

    # Step 2: Collect frames grouped by emotion
    print('==> Collecting frames by emotion from:', args.data_root)
    emotion_frames = collect_frames_by_emotion(args.data_root)

    for emotion, frames in emotion_frames.items():
        print(f'  {emotion}: {len(frames)} frames')

    # Step 3: Extract and save expression weights for each emotion
    os.makedirs(args.save_dir, exist_ok=True)

    for emotion in EMOTION_CATEGORIES:
        frames = emotion_frames[emotion]
        if len(frames) == 0:
            print(f'[WARNING] No frames found for emotion: {emotion}, skipping.')
            continue

        print(f'\n==> Extracting expression weights for: {emotion} ({len(frames)} frames)')
        mean_exp = extract_exp_weights(gen, frames, device, args.max_samples_per_emotion)

        if mean_exp is not None:
            save_path = os.path.join(args.save_dir, f'{emotion}.npy')
            np.save(save_path, mean_exp)
            print(f'  Saved: {save_path} (shape: {mean_exp.shape})')
        else:
            print(f'  [WARNING] Could not extract weights for: {emotion}')

    print('\n==> Done! Expression weights saved to:', args.save_dir)


if __name__ == '__main__':
    main()
