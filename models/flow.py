"""
Normalizing Flow on latent space
=================================
RealNVP-style affine coupling flow.
Maps latent z ↔ N(0,I).

Usage
-----
flow = LatentFlow(latent_dim=128, n_layers=8, hidden_dim=256)

# Training
log_prob = flow.log_prob(z)          # [B]  — maximize during training
loss     = -log_prob.mean()

# Sampling
z_sample = flow.sample(n=100)        # [100, latent_dim]

# Optional: conditional flow  p(z|y)
flow_c = LatentFlow(latent_dim=128, cond_dim=5)
log_prob = flow_c.log_prob(z, cond=y)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Affine coupling layer  (RealNVP)
# ---------------------------------------------------------------------------
class AffineCoupling(nn.Module):
    """
    Split z into (z1, z2).  Transform z2 given z1 (+ optional condition).
    Forward  (z → u):  u2 = z2 * exp(s(z1, cond)) + t(z1, cond)
    Inverse  (u → z):  z2 = (u2 - t(u1, cond)) * exp(-s(u1, cond))
    """

    def __init__(self, dim: int, hidden_dim: int,
                 cond_dim: int = 0, n_hidden: int = 2):
        super().__init__()
        self.split  = dim // 2
        self.dim    = dim
        in_dim = self.split + cond_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.net_s = nn.Sequential(*layers, nn.Linear(hidden_dim, dim - self.split))
        self.net_t = nn.Sequential(*layers[:-1],  # reuse all but last activation
                                   nn.Linear(hidden_dim, dim - self.split))

        # Separate networks for s and t (more expressive)
        self._build(in_dim, hidden_dim, dim - self.split, n_hidden)

        # Learnable scale clamp for stability
        self.log_scale_factor = nn.Parameter(torch.zeros(dim - self.split))

    def _build(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int):
        def make_net(out):
            layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            for _ in range(n_hidden - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, out)]
            return nn.Sequential(*layers)
        self.net_s = make_net(out_dim)
        self.net_t = make_net(out_dim)

    def _st(self, z1: torch.Tensor,
            cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = z1 if cond is None else torch.cat([z1, cond], dim=-1)
        s = self.net_s(inp) * self.log_scale_factor.exp()
        s = torch.tanh(s) * 2.0          # clamp scale to (-2, 2)
        t = self.net_t(inp)
        return s, t

    def forward(self, z: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """z → u,  returns (u, log_det)"""
        z1, z2 = z[:, :self.split], z[:, self.split:]
        s, t   = self._st(z1, cond)
        u2     = z2 * s.exp() + t
        log_det = s.sum(dim=-1)
        return torch.cat([z1, u2], dim=-1), log_det

    def inverse(self, u: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """u → z"""
        u1, u2 = u[:, :self.split], u[:, self.split:]
        s, t   = self._st(u1, cond)
        z2     = (u2 - t) * (-s).exp()
        return torch.cat([u1, z2], dim=-1)


# ---------------------------------------------------------------------------
# Permutation (fixed alternating)
# ---------------------------------------------------------------------------
class Permutation(nn.Module):
    def __init__(self, dim: int, mode: str = 'reverse'):
        super().__init__()
        if mode == 'reverse':
            perm = torch.arange(dim - 1, -1, -1)
        else:
            perm = torch.randperm(dim)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.perm]

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u[:, self.inv_perm]


# ---------------------------------------------------------------------------
# ActNorm (data-dependent initialization)
# ---------------------------------------------------------------------------
class ActNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.loc   = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(dim))
        self._initialized = False

    def _initialize(self, z: torch.Tensor):
        with torch.no_grad():
            self.loc.data   = -z.mean(0)
            self.scale.data  = 1.0 / (z.std(0) + 1e-6)
        self._initialized = True

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized:
            self._initialize(z)
        u       = (z + self.loc) * self.scale
        log_det = self.scale.abs().log().sum()
        return u, log_det.expand(z.size(0))

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u / self.scale - self.loc


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------
class LatentFlow(nn.Module):
    """
    Stack of ActNorm + AffineCoupling + Permutation layers.

    Parameters
    ----------
    latent_dim : dimension of latent z
    n_layers   : number of coupling layers
    hidden_dim : width of coupling networks
    cond_dim   : 0 for unconditional; >0 for conditional p(z|y)
    """

    def __init__(
        self,
        latent_dim:  int   = 128,
        n_layers:    int   = 8,
        hidden_dim:  int   = 256,
        cond_dim:    int   = 0,
        n_hidden:    int   = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

        layers = []
        for i in range(n_layers):
            layers.append(ActNorm(latent_dim))
            layers.append(AffineCoupling(
                latent_dim, hidden_dim, cond_dim=cond_dim, n_hidden=n_hidden))
            if i < n_layers - 1:
                perm_mode = 'reverse' if i % 2 == 0 else 'reverse'
                layers.append(Permutation(latent_dim, mode=perm_mode))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor,
                cond: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """z → u,  returns (u, total_log_det)  shapes: [B,D], [B]"""
        log_det = torch.zeros(z.size(0), device=z.device)
        u = z
        for layer in self.layers:
            if isinstance(layer, AffineCoupling):
                u, ld = layer(u, cond)
                log_det = log_det + ld
            elif isinstance(layer, ActNorm):
                u, ld = layer(u)
                log_det = log_det + ld
            else:  # Permutation
                u = layer(u)
        return u, log_det

    def inverse(self, u: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """u ~ N(0,I) → z"""
        z = u
        for layer in reversed(self.layers):
            if isinstance(layer, AffineCoupling):
                z = layer.inverse(z, cond)
            elif isinstance(layer, ActNorm):
                z = layer.inverse(z)
            else:
                z = layer.inverse(z)
        return z

    def log_prob(self, z: torch.Tensor,
                 cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns log p(z) of shape [B]."""
        u, log_det = self.forward(z, cond)
        # log N(0,I)
        log_base = -0.5 * (self.latent_dim * math.log(2 * math.pi)
                           + (u ** 2).sum(dim=-1))
        return log_base + log_det

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu',
               cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample z ~ p(z) (or p(z|cond) if cond given)."""
        u = torch.randn(n, self.latent_dim, device=device)
        return self.inverse(u, cond)
