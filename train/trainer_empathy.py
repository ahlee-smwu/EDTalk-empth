"""Trainer for the empathetic talking-head model.

Manages the ``EmpathyGenerator``, discriminator, optimizers, and the
combined loss (VGG perceptual + L1 reconstruction + GAN + KL + expression-
matching).
"""

import os
import torch
import torch.nn.functional as F
from torch import nn, optim

from networks.empathy_generator import EmpathyGenerator
from networks.discriminator import Discriminator


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class EmpathyTrainer(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size

        # Build empathy generator
        self.gen = EmpathyGenerator(
            size=args.size,
            style_dim=args.latent_dim_style,
            lip_dim=args.latent_dim_lip,
            pose_dim=args.latent_dim_pose,
            exp_dim=args.latent_dim_exp,
            channel_multiplier=args.channel_multiplier,
            emotion_dim=getattr(args, 'emotion_dim', 128),
            cvae_latent_dim=getattr(args, 'cvae_latent_dim', 32),
        ).to(device)

        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # Optionally load pre-trained backbone weights
        if getattr(args, 'backbone_ckpt', None) is not None:
            print(f'Loading backbone weights from {args.backbone_ckpt}')
            ckpt = torch.load(args.backbone_ckpt, map_location='cpu')
            self.gen.backbone.load_state_dict(ckpt['gen'], strict=False)

        # Freeze backbone by default so only empathy components train
        if getattr(args, 'freeze_backbone', True):
            requires_grad(self.gen.backbone, False)

        # Optimizers ----------------------------------------------------------
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        trainable_params = [
            p for p in self.gen.parameters() if p.requires_grad
        ]
        self.g_optim = optim.Adam(
            trainable_params,
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        # Losses ---------------------------------------------------------------
        from train.vgg19 import VGGLoss
        self.criterion_vgg = VGGLoss().to(device)

        # Loss weights
        self.lambda_kl = getattr(args, 'lambda_kl', 0.01)
        self.lambda_exp = getattr(args, 'lambda_exp', 1.0)

        self.start_iter = 0

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    @staticmethod
    def g_nonsaturating_loss(fake_pred):
        return F.softplus(-fake_pred).mean()

    @staticmethod
    def d_nonsaturating_loss(fake_pred, real_pred):
        return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()

    # ------------------------------------------------------------------
    # Generator update
    # ------------------------------------------------------------------
    def gen_update(self, listener_source, listener_target, speaker_frame,
                   speaker_mel=None):
        self.gen.train()
        self.gen.zero_grad()
        requires_grad(self.gen.speaker_encoder, True)
        requires_grad(self.gen.empathy_cvae, True)
        requires_grad(self.dis, False)

        img_recon, kl_loss, pred_exp, gt_exp = self.gen(
            listener_source, listener_target, speaker_frame, speaker_mel
        )

        # Losses
        vgg_loss = self.criterion_vgg(img_recon, listener_target).mean()
        l1_loss = F.l1_loss(img_recon, listener_target)
        gan_g_loss = self.g_nonsaturating_loss(self.dis(img_recon))
        exp_loss = F.mse_loss(pred_exp, gt_exp.detach())

        g_loss = (
            vgg_loss
            + l1_loss
            + gan_g_loss
            + self.lambda_kl * kl_loss
            + self.lambda_exp * exp_loss
        )
        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, kl_loss, exp_loss, img_recon

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------
    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()
        requires_grad(self.gen.speaker_encoder, False)
        requires_grad(self.gen.empathy_cvae, False)
        requires_grad(self.dis, True)

        real_pred = self.dis(img_real)
        fake_pred = self.dis(img_recon.detach())
        d_loss = self.d_nonsaturating_loss(fake_pred, real_pred)
        d_loss.backward()
        self.d_optim.step()
        return d_loss

    # ------------------------------------------------------------------
    # Evaluation sample
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, listener_source, listener_target, speaker_frame,
               speaker_mel=None):
        self.gen.eval()
        img_recon, kl_loss, pred_exp, gt_exp = self.gen(
            listener_source, listener_target, speaker_frame, speaker_mel
        )
        return img_recon

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------
    def resume(self, resume_ckpt):
        print('Loading checkpoint:', resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location='cpu')
        start_iter = ckpt.get('start_iter', 0)
        self.gen.load_state_dict(ckpt['gen'], strict=False)
        self.dis.load_state_dict(ckpt['dis'], strict=False)
        if 'g_optim' in ckpt:
            self.g_optim.load_state_dict(ckpt['g_optim'])
        if 'd_optim' in ckpt:
            self.d_optim.load_state_dict(ckpt['d_optim'])
        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                'gen': self.gen.state_dict(),
                'dis': self.dis.state_dict(),
                'g_optim': self.g_optim.state_dict(),
                'd_optim': self.d_optim.state_dict(),
                'start_iter': idx,
                'args': self.args,
            },
            os.path.join(checkpoint_path, f'{str(idx).zfill(6)}.pt'),
        )
