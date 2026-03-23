"""Smoke tests for the empathy module components.

Validates tensor shapes and basic forward/backward passes for:
- SpeakerEncoder
- SpeakerAudioEncoder
- EmpathyCVAE
- EmpathyGenerator (training + inference modes)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


def test_speaker_audio_encoder():
    from networks.empathy_module import SpeakerAudioEncoder

    enc = SpeakerAudioEncoder(emotion_dim=128)
    mel = torch.randn(4, 1, 80, 16)
    out = enc(mel)
    assert out.shape == (4, 128), f'Expected (4, 128), got {out.shape}'
    print('[PASS] SpeakerAudioEncoder: output shape', out.shape)


def test_speaker_encoder_video_only():
    from networks.empathy_module import SpeakerEncoder

    enc = SpeakerEncoder(size=256, style_dim=512, emotion_dim=128)
    frame = torch.randn(2, 3, 256, 256)
    out = enc(frame, speaker_mel=None)
    assert out.shape == (2, 128), f'Expected (2, 128), got {out.shape}'
    print('[PASS] SpeakerEncoder (video-only): output shape', out.shape)


def test_speaker_encoder_audio_video():
    from networks.empathy_module import SpeakerEncoder

    enc = SpeakerEncoder(size=256, style_dim=512, emotion_dim=128)
    frame = torch.randn(2, 3, 256, 256)
    mel = torch.randn(2, 1, 80, 16)
    out = enc(frame, mel)
    assert out.shape == (2, 128), f'Expected (2, 128), got {out.shape}'
    print('[PASS] SpeakerEncoder (audio+video): output shape', out.shape)


def test_empathy_cvae_training():
    from networks.empathy_module import EmpathyCVAE

    cvae = EmpathyCVAE(emotion_dim=128, exp_dim=10, latent_dim=32)
    speaker_emotion = torch.randn(4, 128)
    listener_exp_gt = torch.randn(4, 10)

    pred_exp, kl_loss = cvae(speaker_emotion, listener_exp_gt)
    assert pred_exp.shape == (4, 10), f'Expected (4, 10), got {pred_exp.shape}'
    assert kl_loss.item() >= 0, 'KL loss should be non-negative'
    print('[PASS] EmpathyCVAE training: pred shape', pred_exp.shape,
          'kl_loss', kl_loss.item())


def test_empathy_cvae_inference_variety():
    from networks.empathy_module import EmpathyCVAE

    cvae = EmpathyCVAE(emotion_dim=128, exp_dim=10, latent_dim=32)
    cvae.eval()
    speaker_emotion = torch.randn(1, 128)

    results = []
    for _ in range(5):
        pred_exp, kl = cvae(speaker_emotion, listener_exp=None)
        results.append(pred_exp.clone())

    # At least some of the 5 runs should differ (variety)
    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    assert not all_same, 'CVAE inference should produce variety!'
    print('[PASS] EmpathyCVAE inference variety: 5 different outputs confirmed')


def test_empathy_generator_training():
    from networks.empathy_generator import EmpathyGenerator

    gen = EmpathyGenerator(
        size=256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
        channel_multiplier=1, emotion_dim=128, cvae_latent_dim=32,
    )
    gen.train()

    bs = 2
    listener_src = torch.randn(bs, 3, 256, 256)
    listener_tgt = torch.randn(bs, 3, 256, 256)
    speaker_frm = torch.randn(bs, 3, 256, 256)

    img_recon, kl_loss, pred_exp, gt_exp = gen(
        listener_src, listener_tgt, speaker_frm, speaker_mel=None
    )
    assert img_recon.shape == (bs, 3, 256, 256), \
        f'Expected ({bs}, 3, 256, 256), got {img_recon.shape}'
    assert pred_exp.shape == (bs, 10)
    assert gt_exp.shape == (bs, 10)
    print('[PASS] EmpathyGenerator training: recon shape', img_recon.shape)

    # Check backward
    loss = img_recon.mean() + kl_loss
    loss.backward()
    print('[PASS] EmpathyGenerator training: backward pass OK')


def test_empathy_generator_inference_A():
    from networks.empathy_generator import EmpathyGenerator

    gen = EmpathyGenerator(
        size=256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
        channel_multiplier=1, emotion_dim=128, cvae_latent_dim=32,
    )
    gen.eval()

    listener_src = torch.randn(1, 3, 256, 256)
    lip_coeffs = torch.randn(1, 20)
    pose_frame = torch.randn(1, 3, 256, 256)
    speaker_frame = torch.randn(1, 3, 256, 256)

    img = gen.test_empathy_A(
        listener_src, lip_coeffs, pose_frame, speaker_frame
    )
    assert img.shape == (1, 3, 256, 256), \
        f'Expected (1, 3, 256, 256), got {img.shape}'
    print('[PASS] EmpathyGenerator test_empathy_A: output shape', img.shape)


def test_empathy_generator_inference_V():
    from networks.empathy_generator import EmpathyGenerator

    gen = EmpathyGenerator(
        size=256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
        channel_multiplier=1, emotion_dim=128, cvae_latent_dim=32,
    )
    gen.eval()

    listener_src = torch.randn(1, 3, 256, 256)
    lip_frame = torch.randn(1, 3, 256, 256)
    pose_frame = torch.randn(1, 3, 256, 256)
    speaker_frame = torch.randn(1, 3, 256, 256)

    img = gen.test_empathy_V(
        listener_src, lip_frame, pose_frame, speaker_frame
    )
    assert img.shape == (1, 3, 256, 256), \
        f'Expected (1, 3, 256, 256), got {img.shape}'
    print('[PASS] EmpathyGenerator test_empathy_V: output shape', img.shape)


def test_empathy_generator_inference_variety():
    """Verify that multiple inference runs produce different results."""
    from networks.empathy_generator import EmpathyGenerator

    gen = EmpathyGenerator(
        size=256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
        channel_multiplier=1, emotion_dim=128, cvae_latent_dim=32,
    )
    gen.eval()

    listener_src = torch.randn(1, 3, 256, 256)
    lip_coeffs = torch.randn(1, 20)
    pose_frame = torch.randn(1, 3, 256, 256)
    speaker_frame = torch.randn(1, 3, 256, 256)

    results = []
    for _ in range(3):
        img = gen.test_empathy_A(
            listener_src, lip_coeffs, pose_frame, speaker_frame
        )
        results.append(img.clone())

    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    assert not all_same, 'Inference should produce variety (different each run)!'
    print('[PASS] EmpathyGenerator inference variety: confirmed different outputs')


if __name__ == '__main__':
    tests = [
        test_speaker_audio_encoder,
        test_speaker_encoder_video_only,
        test_speaker_encoder_audio_video,
        test_empathy_cvae_training,
        test_empathy_cvae_inference_variety,
        test_empathy_generator_training,
        test_empathy_generator_inference_A,
        test_empathy_generator_inference_V,
        test_empathy_generator_inference_variety,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f'[FAIL] {test_fn.__name__}: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print(f'\n=== {passed} passed, {failed} failed out of {len(tests)} tests ===')
    if failed > 0:
        sys.exit(1)
