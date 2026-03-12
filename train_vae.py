"""
Stage 1: VAE / MAE Pretraining
================================
python train_vae.py --mode vae --epochs 100 --batch 512
python train_vae.py --mode mae --epochs 100 --batch 512 --mask_ratio 0.5
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from dataloader.h5_loader import H5EventDataset
from models.pmtc_vae import PmtCVAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATHS     = ["/home/work/gh-separation/data/gammaD9v40_R100.hdf5"]
POSITION_PATH  = "/home/work/gh-separation/data/position_D9.npz"
CHECKPOINT_DIR = "./checkpoints_vae"
LOG_PATH       = "./logs/train_vae.csv"

# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',       default='vae',  choices=['vae', 'mae'])
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch',      type=int,   default=512)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--latent_dim', type=int,   default=128)
    p.add_argument('--hidden',     type=int,   default=128)
    p.add_argument('--cls_size',   type=int,   default=256)
    p.add_argument('--enc_blocks', type=int,   default=4)
    p.add_argument('--dec_blocks', type=int,   default=2)
    p.add_argument('--beta',       type=float, default=1e-3,
                   help='KL weight for VAE (beta-VAE)')
    p.add_argument('--mask_ratio', type=float, default=0.50,
                   help='Fraction of PMTs masked in MAE mode')
    p.add_argument('--resume',     default=None, help='Checkpoint to resume from')
    p.add_argument('--val_frac',   type=float, default=0.05)
    p.add_argument('--workers',    type=int,   default=8)
    p.add_argument('--save_every', type=int,   default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
def make_collate(pmt_pos):
    """Collate fn: returns (x_features [B,722,5], pmt_pos [722,3])"""
    def collate(batch):
        xs = torch.stack([b['x'] for b in batch])   # [B, 722, 5]
        return xs, pmt_pos
    return collate


# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, pmt_pos):
    model.eval()
    tot_recon = tot_kl = tot_total = n = 0
    for x, _ in loader:
        x = x.to(device)
        losses = model.loss(x, pmt_pos)
        B = x.size(0)
        tot_recon += losses['recon'].item() * B
        tot_kl    += losses['kl'].item()    * B
        tot_total += losses['total'].item() * B
        n += B
    return tot_total/n, tot_recon/n, tot_kl/n


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Mode: {args.mode}  |  Device: {device}  |  Latent: {args.latent_dim}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- Dataset ---
    dataset = H5EventDataset(
        file_paths=DATA_PATHS,
        position_path=POSITION_PATH,
        angle_convergence=True)
    n_val  = max(1, int(len(dataset) * args.val_frac))
    n_trn  = len(dataset) - n_val
    trn_ds, val_ds = random_split(
        dataset, [n_trn, n_val],
        generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_trn}  |  Val: {n_val}")

    # PMT positions [npmts, 3]
    pos_data = np.load(POSITION_PATH)
    pmt_pos = torch.tensor(
        np.stack([pos_data['xpmts'], pos_data['ypmts'], pos_data['zpmts']], axis=-1),
        dtype=torch.float32).to(device)
    pos_data.close()
    npmts = pmt_pos.size(0)

    collate = make_collate(pmt_pos)
    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate)

    # --- Model ---
    model = PmtCVAE(
        npmts=npmts,
        hidden_size=args.hidden,
        classifier_size=args.cls_size,
        latent_dim=args.latent_dim,
        enc_nblocks=args.enc_blocks,
        dec_nblocks=args.dec_blocks,
        mode=args.mode,
        mask_ratio=args.mask_ratio,
        beta=args.beta,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 0
    best_val    = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # --- Log header ---
    if not os.path.exists(LOG_PATH) or start_epoch == 0:
        with open(LOG_PATH, 'w') as f:
            f.write("epoch,trn_total,trn_recon,trn_kl,val_total,val_recon,val_kl,lr,time\n")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        trn_recon = trn_kl = trn_total = 0
        n = 0

        for x, pos in trn_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(x, pos)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            B = x.size(0)
            trn_total += losses['total'].item() * B
            trn_recon += losses['recon'].item() * B
            trn_kl    += losses['kl'].item()    * B
            n += B

        scheduler.step()
        trn_total /= n;  trn_recon /= n;  trn_kl /= n
        val_total, val_recon, val_kl = validate(model, val_loader, device, pmt_pos)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"[{epoch:03d}] trn: {trn_total:.4f} (recon={trn_recon:.4f} kl={trn_kl:.4f})  "
              f"val: {val_total:.4f} (recon={val_recon:.4f} kl={val_kl:.4f})  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        with open(LOG_PATH, 'a') as f:
            f.write(f"{epoch},{trn_total:.6f},{trn_recon:.6f},{trn_kl:.6f},"
                    f"{val_total:.6f},{val_recon:.6f},{val_kl:.6f},"
                    f"{lr_now:.6e},{elapsed:.1f}\n")

        # Save checkpoint
        ckpt = {'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val': best_val, 'args': vars(args)}

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt"))

        if val_total < best_val:
            best_val = val_total
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  ✓ best saved (val_total={best_val:.4f})")

    print("Training complete.")


if __name__ == '__main__':
    main()
