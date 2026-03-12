"""
Stage 3: Reconstruction (Energy / Direction / Core Position)
=============================================================
Trains a reconstruction head on top of a frozen VAE/MAE encoder.

python train_reco.py --vae_ckpt checkpoints_vae/best_model.pt --epochs 50
python train_reco.py --vae_ckpt checkpoints_vae/best_model.pt --finetune  # unfreeze encoder

Targets
-------
  [0] coreX      (shower core X)
  [1] coreY      (shower core Y)
  [2] u_x        (direction cos component)
  [3] u_y        (direction sin component)
  [4] logEnergy  (log10 energy)

Metrics reported
  - Energy  : σ(ΔlogE) / median |ΔlogE|
  - Angular : median angle [deg] between predicted and true direction
  - Core    : median |Δcore| [m]
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

DATA_PATHS     = ["/home/work/gh-separation/data/gammaD9v40_R100.hdf5"]
POSITION_PATH  = "/home/work/gh-separation/data/position_D9.npz"
CHECKPOINT_DIR = "./checkpoints_reco"
LOG_PATH       = "./logs/train_reco.csv"


# ---------------------------------------------------------------------------
class RecoHead(nn.Module):
    """MLP head: latent z → 5 reconstruction targets."""
    def __init__(self, latent_dim: int, hidden: int = 256, nout: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden,     hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden,     nout))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(pred: torch.Tensor, tgt: torch.Tensor,
                    y_mean: torch.Tensor, y_std: torch.Tensor):
    """Denormalize and compute physics metrics."""
    pred_dn = pred * y_std + y_mean
    tgt_dn  = tgt  * y_std + y_mean

    # Energy
    delta_loge = (pred_dn[:, 4] - tgt_dn[:, 4]).cpu().numpy()
    e_sigma    = delta_loge.std()
    e_median   = np.median(np.abs(delta_loge))

    # Angular (u_x, u_y → direction unit vector, assuming u_z = sqrt(1-u_x²-u_y²))
    def to_unit(ux, uy):
        r2  = (ux**2 + uy**2).clamp(max=1.0)
        uz  = (1.0 - r2).sqrt()
        return torch.stack([ux, uy, uz], dim=-1)

    u_pred = to_unit(pred_dn[:, 2], pred_dn[:, 3])
    u_tgt  = to_unit(tgt_dn[:, 2],  tgt_dn[:, 3])
    cos_th = (u_pred * u_tgt).sum(-1).clamp(-1, 1)
    angle_deg = torch.acos(cos_th) * (180 / 3.14159265)
    ang_median = torch.median(angle_deg).item()

    # Core
    delta_core = ((pred_dn[:, 0] - tgt_dn[:, 0])**2
                  + (pred_dn[:, 1] - tgt_dn[:, 1])**2).sqrt().cpu().numpy()
    core_median = np.median(delta_core)

    return {'e_sigma': e_sigma, 'e_median': e_median,
            'ang_median': ang_median, 'core_median': core_median}


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',  required=True)
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--batch',     type=int,   default=512)
    p.add_argument('--lr',        type=float, default=3e-4)
    p.add_argument('--head_hidden', type=int, default=256)
    p.add_argument('--finetune',  action='store_true',
                   help='Unfreeze encoder and finetune jointly')
    p.add_argument('--lr_enc',    type=float, default=3e-5,
                   help='Encoder LR when finetuning')
    p.add_argument('--workers',   type=int,   default=8)
    p.add_argument('--val_frac',  type=float, default=0.05)
    p.add_argument('--save_every', type=int,  default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Finetune encoder: {args.finetune}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- Load VAE ---
    ckpt_vae  = torch.load(args.vae_ckpt, map_location=device)
    vae_args  = ckpt_vae['args']
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
    latent_dim = vae_args['latent_dim']
    print(f"VAE loaded (latent_dim={latent_dim})")

    if not args.finetune:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

    # --- Reco head ---
    head = RecoHead(latent_dim, hidden=args.head_hidden, nout=5).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Reco head parameters: {n_params:,}")

    # --- Dataset ---
    dataset = H5EventDataset(
        file_paths=DATA_PATHS, position_path=POSITION_PATH,
        angle_convergence=True)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_trn = len(dataset) - n_val
    trn_ds, val_ds = random_split(dataset, [n_trn, n_val],
                                  generator=torch.Generator().manual_seed(42))

    # Compute label statistics from training split for normalization
    print("Computing label statistics ...")
    tmp_loader = DataLoader(trn_ds, batch_size=4096, shuffle=False, num_workers=4)
    ys = torch.cat([b['y'] for b in tmp_loader])
    y_mean = ys.mean(0).to(device)
    y_std  = ys.std(0).clamp(min=1e-6).to(device)
    del ys, tmp_loader
    print(f"  y_mean={y_mean.cpu().numpy().round(3)}")
    print(f"  y_std ={y_std.cpu().numpy().round(3)}")

    # PMT positions
    pos_data = np.load(POSITION_PATH)
    pmt_pos = torch.tensor(
        np.stack([pos_data['xpmts'], pos_data['ypmts'], pos_data['zpmts']], axis=-1),
        dtype=torch.float32).to(device)
    pos_data.close()

    def collate(batch):
        return (torch.stack([b['x'] for b in batch]),
                torch.stack([b['y'] for b in batch]))

    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate)

    # --- Optimizer ---
    param_groups = [{'params': head.parameters(), 'lr': args.lr}]
    if args.finetune:
        param_groups.append({'params': vae.parameters(), 'lr': args.lr_enc})

    optimizer = AdamW(param_groups, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_val = float('inf')

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as f:
            f.write("epoch,trn_loss,val_loss,"
                    "e_sigma,e_median,ang_median_deg,core_median_m,lr,time\n")

    criterion = nn.HuberLoss(delta=1.0)

    # --- Training loop ---
    for epoch in range(args.epochs):
        if args.finetune:
            vae.train()
        head.train()
        t0 = time.time()
        trn_loss = n = 0

        for x, y in trn_loader:
            x, y = x.to(device), y.to(device)
            y_norm = (y - y_mean) / y_std

            optimizer.zero_grad(set_to_none=True)

            if args.finetune:
                z, mu, _ = vae.encoder(x)
                z_in = mu
            else:
                with torch.no_grad():
                    z_in = vae.encode(x)

            pred = head(z_in)
            loss = criterion(pred, y_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(list(head.parameters()) +
                                     (list(vae.parameters()) if args.finetune else []), 1.0)
            optimizer.step()
            trn_loss += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()
        trn_loss /= n

        # Validation + metrics
        head.eval()
        vae.eval()
        val_loss = 0;  nv = 0
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_norm = (y - y_mean) / y_std
                z_in   = vae.encode(x)
                pred   = head(z_in)
                val_loss += criterion(pred, y_norm).item() * x.size(0)
                nv += x.size(0)
                all_pred.append(pred.cpu())
                all_tgt.append(y_norm.cpu())

        val_loss /= nv
        metrics = compute_metrics(
            torch.cat(all_pred), torch.cat(all_tgt),
            y_mean.cpu(), y_std.cpu())
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"[{epoch:03d}] trn={trn_loss:.4f}  val={val_loss:.4f}  "
              f"E_σ={metrics['e_sigma']:.3f}  E_med={metrics['e_median']:.3f}  "
              f"ang={metrics['ang_median']:.2f}°  core={metrics['core_median']:.1f}m  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        with open(LOG_PATH, 'a') as f:
            f.write(f"{epoch},{trn_loss:.6f},{val_loss:.6f},"
                    f"{metrics['e_sigma']:.4f},{metrics['e_median']:.4f},"
                    f"{metrics['ang_median']:.4f},{metrics['core_median']:.4f},"
                    f"{lr_now:.6e},{elapsed:.1f}\n")

        ckpt = {
            'epoch': epoch, 'head': head.state_dict(),
            'vae': vae.state_dict() if args.finetune else None,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'y_mean': y_mean.cpu(), 'y_std': y_std.cpu(),
            'args': vars(args), 'metrics': metrics,
        }
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best_reco.pt"))
            print(f"  ✓ best reco saved (val={best_val:.4f})")

    print("Reconstruction training complete.")


if __name__ == '__main__':
    main()
