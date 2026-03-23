import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import EqualLinear, EncoderApp


class SpeakerAudioEncoder(nn.Module):
    """Encodes speaker's audio into an emotion-aware embedding.

    Reuses the same Conv2d architecture as Audio2Lip but maps to a
    higher-dimensional emotion feature instead of 20-dim lip coefficients.
    """

    def __init__(self, emotion_dim=128):
        super().__init__()
        from .audio_encoder import Conv2d as AudioConv2d

        self.audio_encoder = nn.Sequential(
            AudioConv2d(1, 32, kernel_size=3, stride=1, padding=1),
            AudioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            AudioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(64, 128, kernel_size=3, stride=3, padding=1),
            AudioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            AudioConv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            AudioConv2d(256, 512, kernel_size=3, stride=1, padding=0),
            AudioConv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.mapping = nn.Linear(512, emotion_dim)
        nn.init.constant_(self.mapping.bias, 0.0)

    def forward(self, mel_input):
        """
        Args:
            mel_input: [bs*T, 1, 80, 16] mel-spectrogram chunks.
        Returns:
            [bs*T, emotion_dim] per-frame speaker audio emotion features.
        """
        x = self.audio_encoder(mel_input).view(mel_input.size(0), -1)
        return self.mapping(x)


class SpeakerEncoder(nn.Module):
    """Encodes the speaker's emotional state from video frames and audio.

    Fuses visual expression features (extracted via the shared EncoderApp +
    exp_fc) with audio emotion features into a single speaker emotion
    embedding.
    """

    def __init__(self, size=256, style_dim=512, emotion_dim=128):
        super().__init__()
        self.visual_encoder = EncoderApp(size, style_dim)
        self.visual_proj = nn.Sequential(
            EqualLinear(style_dim, style_dim),
            EqualLinear(style_dim, emotion_dim),
        )
        self.audio_encoder = SpeakerAudioEncoder(emotion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(emotion_dim * 2, emotion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emotion_dim, emotion_dim),
        )

    def forward(self, speaker_frame, speaker_mel=None):
        """
        Args:
            speaker_frame: [bs, 3, 256, 256] speaker video frame.
            speaker_mel:   [bs, 1, 80, 16] speaker mel-spectrogram chunk,
                           or None for video-only mode.
        Returns:
            speaker_emotion: [bs, emotion_dim] speaker emotion embedding.
        """
        h_vis, _ = self.visual_encoder(speaker_frame)
        vis_feat = self.visual_proj(h_vis)

        if speaker_mel is not None:
            aud_feat = self.audio_encoder(speaker_mel)
            fused = self.fusion(torch.cat([vis_feat, aud_feat], dim=-1))
        else:
            fused = vis_feat

        return fused


class EmpathyCVAE(nn.Module):
    """Conditional Variational Autoencoder for empathy expression generation.

    During **training** the posterior encoder ``q(z | speaker_emotion,
    listener_exp)`` is used to compute the KL loss, while the decoder
    generates listener expression coefficients from ``z`` and the speaker
    emotion.

    During **inference** the prior encoder ``p(z | speaker_emotion)`` provides
    a stochastic ``z`` so that each run produces a *different* empathetic
    response (variety).
    """

    def __init__(self, emotion_dim=128, exp_dim=10, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Prior: p(z | speaker_emotion)
        self.prior_fc = nn.Sequential(
            nn.Linear(emotion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.prior_mu = nn.Linear(64, latent_dim)
        self.prior_logvar = nn.Linear(64, latent_dim)

        # Posterior: q(z | speaker_emotion, listener_exp)
        self.posterior_fc = nn.Sequential(
            nn.Linear(emotion_dim + exp_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.posterior_mu = nn.Linear(64, latent_dim)
        self.posterior_logvar = nn.Linear(64, latent_dim)

        # Decoder: p(listener_exp | z, speaker_emotion)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + emotion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, exp_dim),
        )

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, speaker_emotion, listener_exp=None):
        """
        Args:
            speaker_emotion: [bs, emotion_dim] speaker emotion embedding.
            listener_exp:    [bs, exp_dim] ground-truth listener expression
                             coefficients (only during training).
        Returns:
            pred_exp:  [bs, exp_dim] predicted listener expression.
            kl_loss:   scalar KL divergence (0 during inference).
        """
        # Prior
        hp = self.prior_fc(speaker_emotion)
        prior_mu = self.prior_mu(hp)
        prior_logvar = self.prior_logvar(hp)

        if listener_exp is not None:
            # Posterior (training)
            hq = self.posterior_fc(torch.cat([speaker_emotion, listener_exp], dim=-1))
            post_mu = self.posterior_mu(hq)
            post_logvar = self.posterior_logvar(hq)
            z = self._reparameterize(post_mu, post_logvar)
            kl_loss = -0.5 * torch.mean(
                1 + post_logvar - prior_logvar
                - (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp()
            )
        else:
            # Inference: sample from prior
            z = self._reparameterize(prior_mu, prior_logvar)
            kl_loss = torch.tensor(0.0, device=speaker_emotion.device)

        pred_exp = self.decoder(torch.cat([z, speaker_emotion], dim=-1))
        return pred_exp, kl_loss
