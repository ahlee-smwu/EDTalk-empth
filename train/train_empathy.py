"""Training entry-point for the empathetic talking-head model.

Usage example (single GPU)::

    python train/train_empathy.py \\
        --path /data/empathy_lmdb \\
        --backbone_ckpt ckpts/EDTalk.pt \\
        --exp_path ./experiments \\
        --exp_name empathy_v1

The script mirrors the structure of ``train/train_E_G.py`` so that the
workflow is familiar.
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from datasets.dataset_empathy import EmpathyDataset
from train.trainer_empathy import EmpathyTrainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    writer.add_images(tag=name, global_step=idx, img_tensor=img)


def write_loss(i, losses, writer):
    for name, val in losses.items():
        writer.add_scalar(name, val.item() if hasattr(val, 'item') else val, i)
    writer.flush()


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = os.path.join(args.exp_path, args.exp_name, 'log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name, 'checkpoint')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Dataset ---------------------------------------------------------------
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    dataset = EmpathyDataset(args, is_inference=False, transform=transform)
    dataset_test = EmpathyDataset(args, is_inference=True, transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    # Trainer ---------------------------------------------------------------
    trainer = EmpathyTrainer(args, device)

    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print(f'Resumed from iteration {args.start_iter}')

    current_iter = args.start_iter
    last_name = None
    best_loss = float('inf')

    print('==> training empathy model')
    for epoch in range(args.epoch):
        for batch in dataloader:
            current_iter += 1

            listener_src = batch['listener_source'].to(device)
            listener_tgt = batch['listener_target'].to(device)
            speaker_frm = batch['speaker_frame'].to(device)

            # Generator update
            (vgg_loss, l1_loss, gan_g_loss,
             kl_loss, exp_loss, img_recon) = trainer.gen_update(
                listener_src, listener_tgt, speaker_frm
            )

            # Discriminator update
            gan_d_loss = trainer.dis_update(listener_tgt, img_recon)

            # Logging
            if current_iter % args.log_iter == 0:
                write_loss(current_iter, {
                    'vgg_loss': vgg_loss, 'l1_loss': l1_loss,
                    'gan_g_loss': gan_g_loss, 'kl_loss': kl_loss,
                    'exp_loss': exp_loss, 'gan_d_loss': gan_d_loss,
                }, writer)

            if current_iter % args.display_freq == 0:
                print(
                    f'[Iter {current_iter}/{args.iter}] '
                    f'vgg={vgg_loss.item():.4f} l1={l1_loss.item():.4f} '
                    f'g={gan_g_loss.item():.4f} d={gan_d_loss.item():.4f} '
                    f'kl={kl_loss.item():.4f} exp={exp_loss.item():.4f}'
                )

            if current_iter % args.image_save_iter == 0:
                sample = F.interpolate(
                    torch.cat([listener_src, listener_tgt, speaker_frm,
                               img_recon.detach()], dim=0),
                    256,
                )
                utils.save_image(
                    sample,
                    os.path.join(
                        checkpoint_path,
                        f'epoch_{epoch:05d}_step_{current_iter:05d}_train.jpg',
                    ),
                    nrow=args.batch_size, normalize=True, range=(-1, 1),
                )

            # Evaluation
            if current_iter % args.eval_iter == 0:
                for bi, test_batch in enumerate(test_dataloader):
                    with torch.no_grad():
                        lis_src = test_batch['listener_source'].to(device)
                        lis_tgt = test_batch['listener_target'].to(device)
                        spk = test_batch['speaker_frame'].to(device)
                        img_recon_test = trainer.sample(lis_src, lis_tgt, spk)

                        sample = F.interpolate(
                            torch.cat([lis_src, lis_tgt, spk,
                                       img_recon_test.detach()], dim=0),
                            256,
                        )
                        fname = os.path.join(
                            checkpoint_path,
                            f'epoch_{epoch:05d}_step_{current_iter:05d}_test.jpg',
                        )
                        utils.save_image(
                            sample, fname,
                            nrow=args.batch_size, normalize=True, range=(-1, 1),
                        )
                        last_name = fname
                        break

            if current_iter % args.save_freq == 0:
                trainer.save(current_iter, checkpoint_path)
                if last_name is not None:
                    shutil.copy(
                        last_name,
                        os.path.join(checkpoint_path,
                                     f'step_{current_iter:06d}.jpg'),
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Iteration / epochs
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--epoch', type=int, default=100)
    # Model dims
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--latent_dim_style', type=int, default=512)
    parser.add_argument('--latent_dim_lip', type=int, default=20)
    parser.add_argument('--latent_dim_pose', type=int, default=6)
    parser.add_argument('--latent_dim_exp', type=int, default=10)
    parser.add_argument('--emotion_dim', type=int, default=128)
    parser.add_argument('--cvae_latent_dim', type=int, default=32)
    # Training hyper-params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--lambda_exp', type=float, default=1.0)
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    # Paths
    parser.add_argument('--path', type=str, default='empathy_lmdb')
    parser.add_argument('--backbone_ckpt', type=str, default='ckpts/EDTalk.pt')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--exp_path', type=str, default='./experiments')
    parser.add_argument('--exp_name', type=str, default='empathy_v1')
    # Data
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    # Logging
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--display_freq', type=int, default=100)
    parser.add_argument('--image_save_iter', type=int, default=1000)
    parser.add_argument('--eval_iter', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=3000)
    parser.add_argument('--start_iter', type=int, default=0)

    args = parser.parse_args()
    main(args)
