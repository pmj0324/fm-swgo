"""
PmtC-VAE / PmtC-MAE
====================
Encoder  : PmtCModel backbone → CLS token → latent z
           VAE mode : z ~ N(mu, sigma)  via reparameterization
           MAE mode : z = CLS token directly (no KL)

Decoder  : z → cross-attention over learnable PMT queries → charge+time per PMT
           (queries conditioned on fixed PMT positions)

Usage
-----
model = PmtCVAE(npmts=722, latent_dim=128, mode='vae')
out   = model(x)          # x: [B, 722, 5]  (time,charge,xpmt,ypmt,zpmt)
loss  = model.loss(x)     # returns dict with 'total','recon','kl'
z     = model.encode(x)   # returns z (sampled or mean)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class CBlock(nn.Module):
    """Classifier-attention: query=cls, key/value=PMT embeddings."""
    def __init__(self, xdim: int, cdim: int, factor: int, dropout: float, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(xdim)
        self.mha  = nn.MultiheadAttention(
            embed_dim=cdim, num_heads=n_heads, batch_first=True,
            dropout=dropout, kdim=xdim, vdim=xdim)
        self.combine = nn.Sequential(
            nn.LayerNorm(cdim),
            nn.Linear(cdim, cdim * factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cdim * factor, cdim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        c = c + self.mha(c, x, x, key_padding_mask=mask)[0]
        c = c + self.combine(c)
        return c


class CrossAttentionBlock(nn.Module):
    """Query=decoder tokens, Key/Value=latent broadcast."""
    def __init__(self, qdim: int, kdim: int, n_heads: int, dropout: float, factor: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(qdim)
        self.norm_k = nn.LayerNorm(kdim)
        self.mha    = nn.MultiheadAttention(
            embed_dim=qdim, num_heads=n_heads, batch_first=True,
            dropout=dropout, kdim=kdim, vdim=kdim)
        self.ff = nn.Sequential(
            nn.LayerNorm(qdim),
            nn.Linear(qdim, qdim * factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(qdim * factor, qdim), nn.Dropout(dropout))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n  = self.norm_q(q)
        kv_n = self.norm_k(kv)
        q = q + self.mha(q_n, kv_n, kv_n)[0]
        q = q + self.ff(q)
        return q


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class PmtCEncoder(nn.Module):
    """
    PMT → CLS latent encoder.
    Input  : x  [B, npmts, 5]  channels = (time, charge, xpmt, ypmt, zpmt)
    Output : z  [B, latent_dim]    (VAE: sampled)
             mu [B, latent_dim]
             lv [B, latent_dim]    (log-variance; zero if mode='mae')
    """
    def __init__(
        self,
        npmts:          int   = 722,
        hidden_size:    int   = 128,
        classifier_size:int   = 256,
        latent_dim:     int   = 128,
        nblocks:        int   = 4,
        nheads:         int   = 4,
        factor:         int   = 2,
        dropout:        float = 0.05,
        mode:           str   = 'vae',   # 'vae' | 'mae'
    ):
        super().__init__()
        self.npmts    = npmts
        self.latent_dim = latent_dim
        self.mode     = mode

        # Separate embeddings for (time,charge) vs (x,y,z)
        self.emb_tc  = nn.Sequential(
            nn.Linear(2,           hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))
        self.emb_xyz = nn.Sequential(
            nn.Linear(3,           hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, classifier_size))

        # Encoder blocks (CLS cross-attends to PMT tokens)
        hdim = hidden_size  # PMT token dim after combine
        self.pmt_combine = nn.Linear(hidden_size * 2, hdim)   # concat tc+xyz → hdim

        # per-block: PMT self-update + CBlock
        self.enc_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hdim + classifier_size, factor * hdim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(factor * hdim, hdim))
            for _ in range(nblocks)])
        self.enc_cblocks = nn.ModuleList([
            CBlock(hdim, classifier_size, factor, dropout, nheads)
            for _ in range(nblocks)])

        # Projection to latent
        if mode == 'vae':
            self.proj_mu  = nn.Linear(classifier_size, latent_dim)
            self.proj_lv  = nn.Linear(classifier_size, latent_dim)
        else:  # mae
            self.proj_z   = nn.Linear(classifier_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, npmts, 5]  → (z, mu, log_var)"""
        B = x.size(0)

        x_tc  = self.emb_tc(x[:, :, :2])               # [B, npmts, H]
        x_xyz = self.emb_xyz(x[:, :, 2:])              # [B, npmts, H]
        x_pmt = self.pmt_combine(
            torch.cat([x_tc, x_xyz], dim=-1))           # [B, npmts, H]

        c = self.cls_token.expand(B, -1, -1)            # [B, 1, C]
        for transform, cblock in zip(self.enc_transform, self.enc_cblocks):
            x_pmt = x_pmt + transform(
                torch.cat([x_pmt, c.expand(-1, self.npmts, -1)], dim=-1))
            c = cblock(x_pmt, c)

        cls = c.squeeze(1)                              # [B, C]

        if self.mode == 'vae':
            mu = self.proj_mu(cls)                      # [B, latent_dim]
            lv = self.proj_lv(cls).clamp(-10., 4.)      # [B, latent_dim]
            z  = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        else:
            z  = self.proj_z(cls)
            mu = z
            lv = torch.zeros_like(z)

        return z, mu, lv


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class PmtCDecoder(nn.Module):
    """
    Latent z → PMT reconstruction.
    Uses fixed PMT positions as query conditioning + cross-attention to z.
    Output: [B, npmts, 2]  (charge, time)
    """
    def __init__(
        self,
        npmts:       int   = 722,
        latent_dim:  int   = 128,
        hidden_size: int   = 128,
        nblocks:     int   = 2,
        nheads:      int   = 4,
        factor:      int   = 2,
        dropout:     float = 0.05,
    ):
        super().__init__()
        self.npmts = npmts

        # Learnable per-PMT query tokens (initialized with positional info)
        self.pmt_queries = nn.Parameter(torch.randn(npmts, hidden_size))

        # Position embedding for queries
        self.pos_emb = nn.Sequential(
            nn.Linear(3, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))

        # Latent projection (z → key/value context for cross-attention)
        self.z_proj = nn.Linear(latent_dim, hidden_size)

        # Cross-attention blocks: queries attend to latent
        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_size, hidden_size, nheads, dropout, factor)
            for _ in range(nblocks)])

        # Output head
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * factor), nn.GELU(),
            nn.Linear(hidden_size * factor, 2))   # → (charge, time)

    def forward(self, z: torch.Tensor,
                pmt_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z       : [B, latent_dim]
        pmt_pos : [B, npmts, 3] or [npmts, 3]  — optional fixed positions
        Returns : [B, npmts, 2]
        """
        B = z.size(0)

        # PMT queries: learnable + optional position bias
        q = self.pmt_queries.unsqueeze(0).expand(B, -1, -1)  # [B, npmts, H]
        if pmt_pos is not None:
            if pmt_pos.dim() == 2:
                pmt_pos = pmt_pos.unsqueeze(0).expand(B, -1, -1)
            q = q + self.pos_emb(pmt_pos)

        # Latent as key/value context  [B, 1, H]
        kv = self.z_proj(z).unsqueeze(1)

        for block in self.ca_blocks:
            q = block(q, kv)

        return self.out_head(q)   # [B, npmts, 2]


# ---------------------------------------------------------------------------
# Full VAE / MAE model
# ---------------------------------------------------------------------------
class PmtCVAE(nn.Module):
    """
    Full VAE or MAE model.

    Parameters
    ----------
    mode        : 'vae' (KL regularized) | 'mae' (masked reconstruction, no KL)
    mask_ratio  : fraction of PMTs to zero-out during MAE (ignored in VAE mode)
    beta        : KL weight for VAE (beta-VAE)
    """

    def __init__(
        self,
        npmts:           int   = 722,
        hidden_size:     int   = 128,
        classifier_size: int   = 256,
        latent_dim:      int   = 128,
        enc_nblocks:     int   = 4,
        dec_nblocks:     int   = 2,
        nheads:          int   = 4,
        factor:          int   = 2,
        dropout:         float = 0.05,
        mode:            str   = 'vae',
        mask_ratio:      float = 0.50,
        beta:            float = 1e-3,
    ):
        super().__init__()
        self.mode       = mode
        self.mask_ratio = mask_ratio
        self.beta       = beta
        self.npmts      = npmts

        self.encoder = PmtCEncoder(
            npmts=npmts, hidden_size=hidden_size,
            classifier_size=classifier_size, latent_dim=latent_dim,
            nblocks=enc_nblocks, nheads=nheads, factor=factor,
            dropout=dropout, mode=mode)
        self.decoder = PmtCDecoder(
            npmts=npmts, latent_dim=latent_dim, hidden_size=hidden_size,
            nblocks=dec_nblocks, nheads=nheads, factor=factor, dropout=dropout)

    # ------------------------------------------------------------------
    def _apply_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly zero out mask_ratio fraction of PMTs. Returns (x_masked, mask)."""
        B, N, C = x.shape
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        n_mask = int(self.mask_ratio * N)
        idx = torch.argsort(torch.rand(B, N, device=x.device), dim=1)[:, :n_mask]
        mask.scatter_(1, idx, True)
        x_masked = x.clone()
        x_masked[mask] = 0.
        return x_masked, mask

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent mean (no sampling). x: [B, npmts, 5]"""
        _, mu, _ = self.encoder(x)
        return mu

    def forward(self, x: torch.Tensor,
                pmt_pos: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        x       : [B, npmts, 5]  (time, charge, xpmt, ypmt, zpmt)
        pmt_pos : [npmts, 3] or None
        Returns dict with keys: z, mu, log_var, recon
        """
        if self.mode == 'mae':
            x_in, _ = self._apply_mask(x)
        else:
            x_in = x

        z, mu, lv = self.encoder(x_in)
        recon = self.decoder(z, pmt_pos)          # [B, npmts, 2]
        return {'z': z, 'mu': mu, 'log_var': lv, 'recon': recon}

    def loss(self, x: torch.Tensor,
             pmt_pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        x: [B, npmts, 5]  — target charge = x[:,1], time = x[:,0]
        """
        out = self.forward(x, pmt_pos)

        # Target: (charge, time) — only on hit PMTs (charge > 0)
        tgt_charge = x[:, :, 1]                          # [B, npmts]
        tgt_time   = x[:, :, 0]                          # [B, npmts]
        target     = torch.stack([tgt_charge, tgt_time], dim=-1)  # [B, npmts, 2]

        if self.mode == 'mae':
            # Loss only on ALL PMTs (MAE reconstructs everything from masked input)
            hit_mask = (tgt_charge > 0).float().unsqueeze(-1)     # [B, npmts, 1]
            recon_loss = (F.mse_loss(out['recon'], target, reduction='none')
                          * hit_mask).sum() / (hit_mask.sum() + 1e-8)
        else:
            # VAE: loss on all PMTs
            recon_loss = F.mse_loss(out['recon'], target)

        # KL divergence (VAE only)
        if self.mode == 'vae':
            kl = -0.5 * (1 + out['log_var'] - out['mu'].pow(2)
                         - out['log_var'].exp()).sum(dim=-1).mean()
        else:
            kl = torch.tensor(0., device=x.device)

        total = recon_loss + self.beta * kl
        return {'total': total, 'recon': recon_loss, 'kl': kl,
                'z': out['z'], 'mu': out['mu']}
