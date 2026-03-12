"""
train_flow_test.py
==================
PmtC-DiT Flow Matching for SWGO PMT events — all-in-one single file.

Architecture: PmtC × DiT hybrid as CFM vector field network
  • AdaLN-Zero          (flow-time + optional shower conditioning)
  • PMT self-attention  (PMT끼리 직접 correlation, DiT style)
  • CLS cross-attention (global coherence, PmtC style)
  • OT-CFM loss         (linear path MSE)

Flow variables per PMT: (log1p(charge), t_rel)  — [B, 722, 2]
PMT positions (x, y) : fixed geometry conditioning  — [722, 2]

Usage
-----
python train_flow_test.py                              # defaults
python train_flow_test.py --epochs 50 --batch 128
python train_flow_test.py --conditional                # p(event | shower)
python train_flow_test.py --hidden 256 --blocks 8      # bigger
python train_flow_test.py --epochs 1 --stats_n 5000    # quick smoke test
"""

import os, sys, math, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# tqdm is optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── repo dataloader ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader.h5_loader import H5EventDataset

# ══════════════════════════════════════════════════════════════════════════════
# 1. ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="PmtC-DiT Flow Matching — SWGO")

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_paths", nargs="+",
        default=["/home/work/gh-separation/data/gammaD9v40_R100.hdf5"])
    p.add_argument("--pos_path",
        default="/home/work/gh-separation/data/position_D9.npz")
    p.add_argument("--val_frac",   type=float, default=0.05)
    p.add_argument("--cut_nhit",   type=float, default=None)
    p.add_argument("--cut_mccorer",type=float, default=None)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--hidden",     type=int,   default=128,
                   help="PMT token hidden dim (also CLS dim)")
    p.add_argument("--cond_dim",   type=int,   default=256,
                   help="flow-time + condition embedding dim")
    p.add_argument("--blocks",     type=int,   default=4)
    p.add_argument("--heads",      type=int,   default=4)
    p.add_argument("--factor",     type=int,   default=2,  help="FFN expansion")
    p.add_argument("--dropout",    type=float, default=0.05)
    p.add_argument("--conditional",action="store_true",
                   help="conditional p(event | coreX,Y,ux,uy,logE)")

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float,default=1e-5)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)

    # ── normalization ─────────────────────────────────────────────────────────
    p.add_argument("--stats_n",    type=int,   default=50000,
                   help="events used to compute normalisation stats")
    p.add_argument("--stats_path", default=None,
                   help="cache path for norm stats (auto if None)")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--out_dir",    default="./checkpoints_fm")
    p.add_argument("--save_every", type=int,   default=5)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_norm_stats(dataset, n_events=50000, batch_size=512, workers=0):
    """
    Compute mean/std of (log1p_q, t_rel) over ALL 722 PMTs × sampled events.
    Includes zero entries (no-hit PMTs) so normalisation is consistent.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)
    log_qs, t_rels = [], []
    n_seen = 0
    print(f"  Computing norm stats from {n_events:,} events ...")
    for batch in loader:
        x = batch["x"]                         # [B, 722, 5]
        charge = x[:, :, 1]                    # [B, 722]
        time_  = x[:, :, 0]                    # [B, 722]
        hit    = (charge > 0).float()

        # log1p charge
        log_q = torch.log1p(charge)            # [B, 722]

        # charge-weighted reference time
        q_sum  = charge.sum(1, keepdim=True).clamp(min=1e-6)
        t_cw   = (charge * time_).sum(1, keepdim=True) / q_sum
        t_rel  = (time_ - t_cw) * hit         # [B, 722], 0 for no-hit

        log_qs.append(log_q.reshape(-1).float())
        t_rels.append(t_rel.reshape(-1).float())
        n_seen += x.size(0)
        if n_seen >= n_events:
            break

    lq = torch.cat(log_qs)
    tr = torch.cat(t_rels)
    stats = dict(
        log_q_mean = float(lq.mean()),
        log_q_std  = float(lq.std().clamp(min=1e-6)),
        t_rel_mean = float(tr.mean()),
        t_rel_std  = float(tr.std().clamp(min=1e-6)),
    )
    print(f"  log_q: mean={stats['log_q_mean']:.4f}  std={stats['log_q_std']:.4f}")
    print(f"  t_rel: mean={stats['t_rel_mean']:.4f}  std={stats['t_rel_std']:.4f}")
    return stats


def preprocess_batch(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    x: [B, 722, 5]  (time, charge, xpmt, ypmt, zpmt)
    Returns x_norm: [B, 722, 2]  normalised (log_q, t_rel)
    """
    charge = x[:, :, 1]
    time_  = x[:, :, 0]
    hit    = (charge > 0).float()

    log_q  = torch.log1p(charge)
    q_sum  = charge.sum(1, keepdim=True).clamp(min=1e-6)
    t_cw   = (charge * time_).sum(1, keepdim=True) / q_sum
    t_rel  = (time_ - t_cw) * hit

    log_q_n = (log_q - stats["log_q_mean"]) / stats["log_q_std"]
    t_rel_n = (t_rel - stats["t_rel_mean"]) / stats["t_rel_std"]

    return torch.stack([log_q_n, t_rel_n], dim=-1)   # [B, 722, 2]


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

# ── 3-a. Sinusoidal timestep embedding ────────────────────────────────────────
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for τ ∈ [0,1], followed by 2-layer MLP."""
    def __init__(self, out_dim: int, max_period: int = 10000):
        super().__init__()
        half = out_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / (half - 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # t: [B]
        args = t[:, None].float() * self.freqs[None, :]   # [B, half]
        emb  = torch.cat([args.cos(), args.sin()], dim=-1) # [B, out_dim]
        return self.mlp(emb)


# ── 3-b. Adaptive LayerNorm (AdaLN-Zero) ──────────────────────────────────────
class AdaLN(nn.Module):
    """
    AdaLN-Zero: out = (1 + γ) * LN(x) + β
    γ, β are predicted from condition c.
    Zero-init ensures identity at start of training.
    """
    def __init__(self, x_dim: int, c_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim, elementwise_affine=False)
        self.proj = nn.Linear(c_dim, 2 * x_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, N, x_dim],  c: [B, c_dim]
        gamma, beta = self.proj(c).chunk(2, dim=-1)       # [B, x_dim] each
        return (1 + gamma.unsqueeze(1)) * self.norm(x) + beta.unsqueeze(1)


# ── 3-c. PmtDiT Block ─────────────────────────────────────────────────────────
class PmtDiTBlock(nn.Module):
    """
    Hybrid PmtC-DiT block:
      1. AdaLN + PMT self-attention   (DiT style — PMT끼리 직접 소통)
      2. CLS cross-attends to PMTs    (PmtC style — global summary update)
      3. PMTs cross-attend to CLS     (global info broadcast back to PMTs)
      4. AdaLN + FFN
    """
    def __init__(self, dim: int, cond_dim: int, n_heads: int,
                 factor: int, dropout: float):
        super().__init__()

        # 1. PMT self-attention
        self.adaLN_sa  = AdaLN(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)

        # 2. CLS cross-attends to PMTs
        self.norm_cls1  = nn.LayerNorm(dim)
        self.norm_pmt_k = nn.LayerNorm(dim)
        self.cls_attn   = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.cls_ff     = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * factor, dim), nn.Dropout(dropout))

        # 3. PMTs cross-attend to CLS (broadcast)
        self.norm_pmt_bq = nn.LayerNorm(dim)
        self.norm_cls2   = nn.LayerNorm(dim)
        self.broad_attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)

        # 4. FFN
        self.adaLN_ff = AdaLN(dim, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * factor, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, cls: torch.Tensor,
                c: torch.Tensor) -> tuple:
        """
        x   : [B, 722, dim]
        cls : [B,   1, dim]
        c   : [B, cond_dim]
        """
        # 1. PMT self-attention (AdaLN conditioned)
        x_ln = self.adaLN_sa(x, c)
        x    = x + self.self_attn(x_ln, x_ln, x_ln)[0]

        # 2. CLS update: attends to PMTs
        cls = cls + self.cls_attn(
            self.norm_cls1(cls),
            self.norm_pmt_k(x),
            self.norm_pmt_k(x))[0]
        cls = cls + self.cls_ff(cls)

        # 3. PMTs attend to CLS (broadcast global context)
        x = x + self.broad_attn(
            self.norm_pmt_bq(x),
            self.norm_cls2(cls),
            self.norm_cls2(cls))[0]

        # 4. FFN (AdaLN conditioned)
        x = x + self.ff(self.adaLN_ff(x, c))

        return x, cls


# ── 3-d. Full vector field network ────────────────────────────────────────────
class PmtCDiTFM(nn.Module):
    """
    PmtC-DiT Flow Matching vector field network.

    Input:
      x_t   : [B, 722, 2]  current (noisy/interpolated) event
      tau   : [B]           flow time ∈ [0, 1]
      pmt_pos:[722, 2]      fixed PMT (x,y) positions
      y     : [B, 5] or None  shower params (conditional mode)

    Output:
      v     : [B, 722, 2]  predicted velocity field
    """
    def __init__(self, dim: int = 128, cond_dim: int = 256,
                 n_blocks: int = 4, n_heads: int = 4,
                 factor: int = 2, dropout: float = 0.05,
                 conditional: bool = False, cond_in: int = 5):
        super().__init__()
        self.conditional = conditional

        # Signal embedding: (log_q, t_rel) → dim
        self.signal_emb = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(),
            nn.Linear(dim, dim))

        # Position embedding: fixed PMT (x,y) → dim
        self.pos_emb = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(),
            nn.Linear(dim, dim))

        # Flow time embedding → cond_dim
        self.time_emb = TimestepEmbedding(cond_dim)

        # Optional shower condition → cond_dim
        if conditional:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, cond_dim), nn.SiLU(),
                nn.Linear(cond_dim, cond_dim))

        # CLS token: learned init + condition projection
        self.cls_token    = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_cond_proj = nn.Linear(cond_dim, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PmtDiTBlock(dim, cond_dim, n_heads, factor, dropout)
            for _ in range(n_blocks)])

        # Output head: AdaLN-Zero + Linear → velocity [B, 722, 2]
        self.out_adaLN = AdaLN(dim, cond_dim)
        self.out_head  = nn.Linear(dim, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x_t: torch.Tensor, tau: torch.Tensor,
                pmt_pos: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        B = x_t.size(0)

        # ── condition vector ────────────────────────────────────────────────
        c = self.time_emb(tau)                             # [B, cond_dim]
        if self.conditional and y is not None:
            c = c + self.cond_mlp(y)

        # ── PMT tokens ──────────────────────────────────────────────────────
        # pmt_pos: [722, 2] → broadcast to [B, 722, dim]
        pos_enc = self.pos_emb(pmt_pos).unsqueeze(0)      # [1, 722, dim]
        x = self.signal_emb(x_t) + pos_enc                # [B, 722, dim]

        # ── CLS token ───────────────────────────────────────────────────────
        cls = (self.cls_token.expand(B, -1, -1)
               + self.cls_cond_proj(c).unsqueeze(1))      # [B, 1, dim]

        # ── Transformer blocks ──────────────────────────────────────────────
        for block in self.blocks:
            x, cls = block(x, cls, c)

        # ── Output velocity ─────────────────────────────────────────────────
        v = self.out_head(self.out_adaLN(x, c))           # [B, 722, 2]
        return v

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FLOW MATCHING LOSS  (OT-CFM)
# ══════════════════════════════════════════════════════════════════════════════
def cfm_loss(model: nn.Module, x: torch.Tensor,
             pmt_pos: torch.Tensor, stats: dict,
             conditional: bool = False) -> torch.Tensor:
    """
    Optimal Transport Conditional Flow Matching loss.
      x: [B, 722, 5]  raw batch from dataloader  (time, charge, xpmt, ypmt, zpmt, [label])
    """
    device = x.device
    B = x.size(0)

    # Raw features and labels
    feats = x[:, :, :5]     # [B, 722, 5]

    # Shower labels (for conditional mode)
    y_cond = None
    if conditional:
        # Extract label from batch — passed via collate; here we skip for now
        # y_cond = y  (handled in training loop)
        pass

    # Normalise to flow space
    x1 = preprocess_batch(feats, stats).to(device)  # [B, 722, 2]

    # Sample noise and flow time
    x0  = torch.randn_like(x1)                      # [B, 722, 2]
    tau = torch.rand(B, device=device)              # [B]

    # Linear interpolation (OT path)
    t   = tau[:, None, None]
    x_t = (1.0 - t) * x0 + t * x1                  # [B, 722, 2]
    u_t = x1 - x0                                   # target velocity (constant)

    # Predict velocity
    v   = model(x_t, tau, pmt_pos, y_cond)          # [B, 722, 2]

    return F.mse_loss(v, u_t)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, pmt_pos, stats, optimizer,
              device, grad_clip, conditional, train=True):
    model.train(train)
    total_loss = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            feats = batch["x"].to(device)   # [B, 722, 5]
            y     = batch["y"].to(device)   # [B, 5]

            # ── normalise ──────────────────────────────────────────────────
            x1  = preprocess_batch(feats, stats)         # [B, 722, 2]
            B   = x1.size(0)
            x0  = torch.randn_like(x1)
            tau = torch.rand(B, device=device)
            t   = tau[:, None, None]
            x_t = (1.0 - t) * x0 + t * x1
            u_t = x1 - x0

            y_cond = y if conditional else None
            v      = model(x_t, tau, pmt_pos, y_cond)
            loss   = F.mse_loss(v, u_t)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * B
            n += B

    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args    = get_args()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print(f"  PmtC-DiT Flow Matching  |  device: {device}")
    print(f"  hidden={args.hidden}  cond_dim={args.cond_dim}"
          f"  blocks={args.blocks}  heads={args.heads}")
    print(f"  conditional={args.conditional}")
    print("=" * 60)

    # ── PMT positions ──────────────────────────────────────────────────────────
    pos_data = np.load(args.pos_path)
    pmt_pos  = torch.tensor(
        np.stack([pos_data["xpmts"], pos_data["ypmts"]], axis=-1),
        dtype=torch.float32).to(device)              # [722, 2]
    pos_data.close()
    print(f"PMT positions: {pmt_pos.shape}  x∈[{pmt_pos[:,0].min():.1f},{pmt_pos[:,0].max():.1f}]")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from:\n  {args.data_paths}")
    dataset = H5EventDataset(
        file_paths     = args.data_paths,
        position_path  = args.pos_path,
        angle_convergence = True,
        cut_nhit       = args.cut_nhit,
        cut_mccorer    = args.cut_mccorer,
    )
    n_total = len(dataset)
    n_val   = max(1, int(n_total * args.val_frac))
    n_trn   = n_total - n_val
    trn_ds, val_ds = random_split(
        dataset, [n_trn, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    print(f"  train: {n_trn:,}   val: {n_val:,}")

    trn_loader = DataLoader(trn_ds, batch_size=args.batch,  shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch*2, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    # ── Normalisation stats ────────────────────────────────────────────────────
    stats_path = args.stats_path or os.path.join(args.out_dir, "norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\nLoaded norm stats from {stats_path}")
        print(f"  log_q: mean={stats['log_q_mean']:.4f}  std={stats['log_q_std']:.4f}")
        print(f"  t_rel: mean={stats['t_rel_mean']:.4f}  std={stats['t_rel_std']:.4f}")
    else:
        print("\nComputing norm stats ...")
        stats = compute_norm_stats(
            dataset, n_events=args.stats_n,
            batch_size=512, workers=min(args.workers, 2))
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to {stats_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PmtCDiTFM(
        dim         = args.hidden,
        cond_dim    = args.cond_dim,
        n_blocks    = args.blocks,
        n_heads     = args.heads,
        factor      = args.factor,
        dropout     = args.dropout,
        conditional = args.conditional,
    ).to(device)
    n_params = model.count_params()
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ──────────────────────────────────────────────────────────
    log_path = os.path.join("logs", "train_fm.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,trn_loss,val_loss,lr,secs\n")

    best_val = float("inf")
    print(f"\n{'Epoch':>5}  {'trn_loss':>10}  {'val_loss':>10}  {'lr':>10}  {'secs':>6}")
    print("-" * 52)

    for epoch in range(args.epochs):
        t0 = time.time()

        trn_loss = run_epoch(model, trn_loader, pmt_pos, stats,
                             optimizer, device, args.grad_clip,
                             args.conditional, train=True)
        val_loss = run_epoch(model, val_loader, pmt_pos, stats,
                             None, device, 0.0,
                             args.conditional, train=False)
        scheduler.step()
        secs = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"{epoch:>5}  {trn_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{lr_now:>10.2e}  {secs:>6.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{trn_loss:.6f},{val_loss:.6f},{lr_now:.6e},{secs:.1f}\n")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "stats": stats,
            "val_loss": val_loss,
        }
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_ep{epoch:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"       ✓ best saved  (val={best_val:.5f})")

    print(f"\nDone.  Best val loss: {best_val:.5f}")
    print(f"Checkpoints in: {args.out_dir}")


if __name__ == "__main__":
    main()
