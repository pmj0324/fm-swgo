"""
Stage 2: Normalizing Flow on Latent Space
==========================================
Trains a RealNVP flow on z samples from a frozen VAE/MAE encoder.

python train_flow.py --vae_ckpt checkpoints_vae/best_model.pt --epochs 50
python train_flow.py --vae_ckpt checkpoints_vae/best_model.pt --conditional  # p(z|y)
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
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(__file__))
from dataloader.h5_loader import H5EventDataset
from models.pmtc_vae import PmtCVAE
from models.flow import LatentFlow

DATA_PATHS     = ["/home/work/gh-separation/data/gammaD9v40_R100.hdf5"]
POSITION_PATH  = "/home/work/gh-separation/data/position_D9.npz"
CHECKPOINT_DIR = "./checkpoints_flow"
LOG_PATH       = "./logs/train_flow.csv"

LABEL_NAMES    = ['coreX', 'coreY', 'u_x', 'u_y', 'logEnergy']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',    required=True, help='Path to VAE/MAE checkpoint')
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch',       type=int,   default=1024)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--n_layers',    type=int,   default=8,
                   help='Number of coupling layers in flow')
    p.add_argument('--hidden_dim',  type=int,   default=256)
    p.add_argument('--conditional', action='store_true',
                   help='Train conditional flow p(z|y) instead of marginal p(z)')
    p.add_argument('--workers',     type=int,   default=8)
    p.add_argument('--save_every',  type=int,   default=10)
    p.add_argument('--val_frac',    type=float, default=0.05)
    p.add_argument('--cache_latents', action='store_true',
                   help='Pre-compute all latents before training (faster, needs RAM)')
    return p.parse_args()


# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_latents(vae, loader, device, pmt_pos):
    """Run encoder on full dataset, return (z_all [N,D], y_all [N,5])."""
    vae.eval()
    zs, ys = [], []
    for batch in loader:
        x = batch['x'].to(device)         # [B, 722, 5]
        y = batch['y'].to(device)         # [B, 5]
        mu = vae.encode(x)                # [B, latent_dim]  (no sampling = mean)
        zs.append(mu.cpu())
        ys.append(y.cpu())
    return torch.cat(zs), torch.cat(ys)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Conditional: {args.conditional}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- Load VAE ---
    ckpt_vae = torch.load(args.vae_ckpt, map_location=device)
    vae_args = ckpt_vae['args']
    vae = PmtCVAE(
        npmts=722,
        hidden_size=vae_args['hidden'],
        classifier_size=vae_args['cls_size'],
        latent_dim=vae_args['latent_dim'],
        enc_nblocks=vae_args['enc_blocks'],
        dec_nblocks=vae_args['dec_blocks'],
        mode=vae_args['mode'],
    ).to(device)
    vae.load_state_dict(ckpt_vae['model'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latent_dim = vae_args['latent_dim']
    print(f"VAE loaded (latent_dim={latent_dim}, mode={vae_args['mode']})")

    # --- Load dataset ---
    dataset = H5EventDataset(
        file_paths=DATA_PATHS,
        position_path=POSITION_PATH,
        angle_convergence=True)

    raw_loader = DataLoader(dataset, batch_size=1024, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # PMT positions
    pos_data = np.load(POSITION_PATH)
    pmt_pos = torch.tensor(
        np.stack([pos_data['xpmts'], pos_data['ypmts'], pos_data['zpmts']], axis=-1),
        dtype=torch.float32).to(device)
    pos_data.close()

    # Extract / cache latents
    print("Extracting latents ...")
    z_all, y_all = extract_latents(vae, raw_loader, device, pmt_pos)
    print(f"  z: {z_all.shape}, y: {y_all.shape}")

    # Normalize labels for conditional flow
    y_mean = y_all.mean(0)
    y_std  = y_all.std(0).clamp(min=1e-6)
    y_norm = (y_all - y_mean) / y_std

    # Build TensorDataset for fast iteration
    ds = TensorDataset(z_all, y_norm)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_trn = len(ds) - n_val
    trn_ds, val_ds = random_split(ds, [n_trn, n_val],
                                  generator=torch.Generator().manual_seed(42))
    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                            num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=0, pin_memory=False)

    # --- Flow model ---
    cond_dim = y_all.size(1) if args.conditional else 0
    flow = LatentFlow(
        latent_dim=latent_dim,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"Flow parameters: {n_params:,}")

    optimizer = AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    best_val = float('inf')

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as f:
            f.write("epoch,trn_nll,val_nll,lr,time\n")

    # --- Training loop ---
    for epoch in range(args.epochs):
        flow.train()
        t0 = time.time()
        trn_nll = n = 0

        for z, y in trn_loader:
            z = z.to(device)
            cond = y.to(device) if args.conditional else None
            optimizer.zero_grad(set_to_none=True)
            log_prob = flow.log_prob(z, cond)
            loss = -log_prob.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            trn_nll += loss.item() * z.size(0)
            n += z.size(0)

        scheduler.step()
        trn_nll /= n

        # Validation
        flow.eval()
        val_nll = 0; nv = 0
        with torch.no_grad():
            for z, y in val_loader:
                z = z.to(device)
                cond = y.to(device) if args.conditional else None
                lp = flow.log_prob(z, cond)
                val_nll += (-lp).sum().item()
                nv += z.size(0)
        val_nll /= nv
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"[{epoch:03d}] trn_nll={trn_nll:.4f}  val_nll={val_nll:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        with open(LOG_PATH, 'a') as f:
            f.write(f"{epoch},{trn_nll:.6f},{val_nll:.6f},{lr_now:.6e},{elapsed:.1f}\n")

        ckpt = {
            'epoch': epoch, 'model': flow.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'args': vars(args), 'y_mean': y_mean, 'y_std': y_std,
            'latent_dim': latent_dim, 'cond_dim': cond_dim,
        }
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt"))
        if val_nll < best_val:
            best_val = val_nll
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best_flow.pt"))
            print(f"  ✓ best flow saved (val_nll={best_val:.4f})")

    print("Flow training complete.")


if __name__ == '__main__':
    main()
