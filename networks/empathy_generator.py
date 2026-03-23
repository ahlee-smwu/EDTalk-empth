import torch
import torch.nn as nn
from .generator import Generator
from .empathy_module import SpeakerEncoder, EmpathyCVAE


class EmpathyGenerator(nn.Module):
    """Empathetic talking-head generator.

    Wraps the pre-trained EDTalk backbone ``Generator`` with:
    * a ``SpeakerEncoder`` that reads the speaker's video frame (and
      optionally audio) to extract a speaker emotion embedding, and
    * an ``EmpathyCVAE`` that maps that embedding to listener expression
      coefficients with stochastic variety.

    The backbone handles lip-sync, head pose and face rendering as usual;
    only the *expression* channel is replaced by the empathy pathway.
    """

    def __init__(
        self,
        size=256,
        style_dim=512,
        lip_dim=20,
        pose_dim=6,
        exp_dim=10,
        channel_multiplier=1,
        emotion_dim=128,
        cvae_latent_dim=32,
    ):
        super().__init__()
        self.exp_dim = exp_dim
        self.emotion_dim = emotion_dim

        # Pre-trained backbone (weights loaded externally)
        self.backbone = Generator(
            size, style_dim, lip_dim, pose_dim, exp_dim, channel_multiplier
        )

        # New empathy components (trained from scratch)
        self.speaker_encoder = SpeakerEncoder(
            size=size, style_dim=style_dim, emotion_dim=emotion_dim
        )
        self.empathy_cvae = EmpathyCVAE(
            emotion_dim=emotion_dim, exp_dim=exp_dim, latent_dim=cvae_latent_dim
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        listener_source,
        listener_target,
        speaker_frame,
        speaker_mel=None,
    ):
        """Training forward pass.

        Args:
            listener_source: [bs, 3, H, W] listener identity frame.
            listener_target: [bs, 3, H, W] ground-truth listener frame
                             (used to extract GT lip/pose/exp for
                             supervision).
            speaker_frame:   [bs, 3, H, W] speaker video frame.
            speaker_mel:     [bs, 1, 80, 16] speaker mel chunk or None.

        Returns:
            img_recon:  [bs, 3, H, W] reconstructed listener frame.
            kl_loss:    scalar KL divergence from the CVAE.
            pred_exp:   [bs, exp_dim] predicted listener expression coeff.
            gt_exp:     [bs, exp_dim] GT listener expression coeff.
        """
        gen = self.backbone

        # --- Listener encoding (identity + target motion) ----------------
        wa, wa_t, feats, _ = gen.enc(listener_source, listener_target)

        # GT lip / pose / exp from the listener target
        shared_fc_t = gen.fc(wa_t)
        gt_lip = gen.lip_fc(shared_fc_t)
        gt_pose = gen.pose_fc(shared_fc_t)
        gt_exp = gen.exp_fc(shared_fc_t)

        # --- Speaker → empathetic expression ----------------------------
        speaker_emotion = self.speaker_encoder(speaker_frame, speaker_mel)
        pred_exp, kl_loss = self.empathy_cvae(speaker_emotion, gt_exp)

        # --- Render using backbone decoder with predicted expression -----
        alpha_D = torch.cat([gt_lip, gt_pose, pred_exp], dim=-1)
        a = gen.direction_exp.get_shared_out(
            alpha_D, gen.direction_lipnonlip.weight
        )
        e = gen.direction_exp.get_exp_latent(a)
        directions = gen.direction_exp(alpha_D, gen.direction_lipnonlip.weight)
        latent = wa + directions
        img_recon = gen.dec(latent, feats, e)

        return img_recon, kl_loss, pred_exp, gt_exp

    # ------------------------------------------------------------------
    # Inference — audio-driven listener with empathy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def test_empathy_A(
        self,
        listener_source,
        listener_lip_coeffs,
        listener_pose_frame,
        speaker_frame,
        speaker_mel=None,
        h_start=None,
    ):
        """Audio-driven empathetic inference.

        Args:
            listener_source:     [1, 3, H, W] listener identity image.
            listener_lip_coeffs: [1, lip_dim] lip coefficients from Audio2Lip.
            listener_pose_frame: [1, 3, H, W] head-pose reference frame.
            speaker_frame:       [1, 3, H, W] current speaker frame.
            speaker_mel:         [1, 1, 80, 16] speaker mel chunk or None.
            h_start:             optional hidden state (unused, for API compat).

        Returns:
            img_recon: [1, 3, H, W] generated empathetic listener frame.
        """
        gen = self.backbone

        # Listener identity + pose
        wa, wa_t_p, feats, _ = gen.enc(listener_source, listener_pose_frame, h_start)
        alpha_lip = listener_lip_coeffs
        shared_fc_p = gen.fc(wa_t_p)
        alpha_pose = gen.pose_fc(shared_fc_p)

        # Speaker emotion → listener expression
        speaker_emotion = self.speaker_encoder(speaker_frame, speaker_mel)
        alpha_exp, _ = self.empathy_cvae(speaker_emotion)  # stochastic

        # Render
        alpha_D = torch.cat([alpha_lip, alpha_pose, alpha_exp], dim=-1)
        a = gen.direction_exp.get_shared_out(
            alpha_D, gen.direction_lipnonlip.weight
        )
        e = gen.direction_exp.get_exp_latent(a)
        directions = gen.direction_exp(alpha_D, gen.direction_lipnonlip.weight)
        latent = wa + directions
        img_recon = gen.dec(latent, feats, e)
        return img_recon

    # ------------------------------------------------------------------
    # Inference — video-driven listener with empathy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def test_empathy_V(
        self,
        listener_source,
        listener_lip_frame,
        listener_pose_frame,
        speaker_frame,
        speaker_mel=None,
        h_start=None,
    ):
        """Video-driven empathetic inference.

        Args:
            listener_source:     [1, 3, H, W] listener identity image.
            listener_lip_frame:  [1, 3, H, W] lip driving frame.
            listener_pose_frame: [1, 3, H, W] pose driving frame.
            speaker_frame:       [1, 3, H, W] speaker video frame.
            speaker_mel:         [1, 1, 80, 16] speaker mel chunk or None.
            h_start:             optional.

        Returns:
            img_recon: [1, 3, H, W] generated empathetic listener frame.
        """
        gen = self.backbone

        wa, wa_t, feats, _ = gen.enc(listener_source, listener_lip_frame, h_start)
        wa_t_p, _, _, _ = gen.enc(listener_pose_frame, None)

        shared_fc = gen.fc(wa_t)
        alpha_lip = gen.lip_fc(shared_fc)

        shared_fc_p = gen.fc(wa_t_p)
        alpha_pose = gen.pose_fc(shared_fc_p)

        # Speaker emotion → listener expression (stochastic)
        speaker_emotion = self.speaker_encoder(speaker_frame, speaker_mel)
        alpha_exp, _ = self.empathy_cvae(speaker_emotion)

        alpha_D = torch.cat([alpha_lip, alpha_pose, alpha_exp], dim=-1)
        a = gen.direction_exp.get_shared_out(
            alpha_D, gen.direction_lipnonlip.weight
        )
        e = gen.direction_exp.get_exp_latent(a)
        directions = gen.direction_exp(alpha_D, gen.direction_lipnonlip.weight)
        latent = wa + directions
        img_recon = gen.dec(latent, feats, e)
        return img_recon
