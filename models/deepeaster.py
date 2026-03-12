"""
PMTC: CBlock + PmtCModel. Input: [batch, 5, num_pmts] = (t, c, x, y, z) per PMT.
No offset/scale, no label_scale/label_offset, no pmt_offset/pmt_scale.
Optional VAE-style pretraining: encode cls -> z, decode z -> recon [B, 5, npmts].
"""
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

NIN = 5   # time, charge, x_pmt, y_pmt, z_pmt
NCT = 2   # time, charge (core); rest = position (x,y,z)


class CBlock(nn.Module):
    """Classifier-attention block: query=cls, key/value=PMT embeddings."""

    def __init__(self, xdim: int, cdim: int, factor: int, dropout: float, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(xdim)
        self.mha = nn.MultiheadAttention(
            embed_dim=cdim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
            kdim=xdim,
            vdim=xdim,
        )
        self.combine = nn.Sequential(
            nn.LayerNorm(cdim),
            nn.Linear(cdim, cdim * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cdim * factor, cdim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, c: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        c = c + self.mha(c, x, x, key_padding_mask=mask)[0]
        c = c + self.combine(c)
        return c


class PmtCModel(nn.Module):
    """
    PMT Classifier-Attention. Input: [batch, 5, num_pmts] = (time, charge, x_pmt, y_pmt, z_pmt).
    No offset/scale, no label transform, no per-PMT offset/scale.
    """

    def __init__(
        self,
        npmts: int = 722,
        dropout: float = 0.05,
        nblocks: int = 2,
        nheads: int = 2,
        hidden_size: int = 16,
        factor: int = 2,
        final_hidden_size: int = 2048,
        classifier_size: int = 128,
        nout: int = 1,
        final: str = "sigmoid",
        update_with_classifer: bool = True,
        combine: str = "add",
        uncertainty: bool = False,
        uncertainty2: bool = False,
        # VAE-style pretraining
        vae_pretrain: bool = False,
        latent_dim: int = 32,
        vae_decoder_hidden: int = 256,
        vae_beta: float = 1.0,
    ):
        super().__init__()
        self.npmts = npmts
        self.dropout = dropout
        self.nblocks = nblocks
        self.nheads = nheads
        self.hidden_size = hidden_size
        self.factor = factor
        self.classifier_size = classifier_size
        self.nout = nout
        self.update_with_classifer = update_with_classifer
        self.combine = combine.lower()
        self.uncertainty = uncertainty
        self.uncertainty2 = uncertainty2
        self.vae_pretrain = vae_pretrain
        self.latent_dim = latent_dim
        self.vae_beta = vae_beta

        # Embedding: (t, c) and (x, y, z) separately, then combine
        self.embedding_tc = nn.Sequential(
            nn.Linear(NCT, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embedding_xyz = nn.Sequential(
            nn.Linear(NIN - NCT, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        hidden_attn = hidden_size * 2 if self.combine == "concat" else hidden_size

        # Blocks
        transform_in = hidden_attn + classifier_size if update_with_classifer else hidden_attn
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(transform_in, factor * hidden_attn),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(factor * hidden_attn, hidden_attn),
                ),
                CBlock(hidden_attn, classifier_size, factor, dropout, nheads),
            ])
            for _ in range(nblocks)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, classifier_size))
        self.mlp_head = nn.Sequential(
            nn.Linear(classifier_size, final_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size, nout),
        )

        if final == "softmax":
            self.final_fn = lambda x: torch.nn.functional.softmax(x, dim=-1)
        elif final == "identity":
            self.final_fn = nn.Identity()
        elif final == "sigmoid":
            self.final_fn = torch.sigmoid
        else:
            raise AttributeError(f"Unknown final: {final}")

        if uncertainty:
            self.uncertainty_head = nn.Linear(classifier_size + 1, nout)
        if uncertainty2:
            self.uncertainty_head = nn.Linear(classifier_size, nout)

        if vae_pretrain:
            self.fc_mu = nn.Linear(classifier_size, latent_dim)
            self.fc_logvar = nn.Linear(classifier_size, latent_dim)
            self.vae_decoder = nn.Sequential(
                nn.Linear(latent_dim, vae_decoder_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(vae_decoder_hidden, NIN * npmts),
            )

    def transform(self, data) -> None:
        pass

    def transform_label(self, label: Tensor, t_label: Optional[Tensor] = None) -> Tensor:
        return label if t_label is None else t_label

    def transform_output(self, output: Tensor, t_output: Optional[Tensor] = None) -> Tensor:
        return output if t_output is None else t_output

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """z = mu + eps * exp(0.5 * log_var), eps ~ N(0,1)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=mu.device, dtype=mu.dtype)
        return mu + eps * std

    def forward(
        self,
        x: Tensor,
        nHit: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, dict]]:
        # x: [B, 5, npmts] -> [B, npmts, 5]; channels = (t, c, x, y, z)
        x_in = x.transpose(1, 2)
        x_tc = self.embedding_tc(x_in[:, :, :NCT])
        x_xyz = self.embedding_xyz(x_in[:, :, NCT:])
        if self.combine == "concat":
            x = torch.cat([x_tc, x_xyz], dim=-1)
        else:
            x = x_tc + x_xyz

        c = self.cls_token.expand(x.size(0), -1, -1)
        for transform, block in self.blocks:
            if self.update_with_classifer:
                x = x + transform(torch.cat([x, c.expand(-1, self.npmts, -1)], dim=2))
            else:
                x = x + transform(x)
            c = block(x, c)

        vae_aux: Optional[dict] = None
        if self.vae_pretrain:
            c_flat = c.squeeze(1)
            mu = self.fc_mu(c_flat)
            log_var = self.fc_logvar(c_flat)
            z = self.reparameterize(mu, log_var)
            recon = self.vae_decoder(z).view(x.size(0), NIN, self.npmts)
            vae_aux = {"recon": recon, "mu": mu, "log_var": log_var}

        c = self.final_fn(self.mlp_head(c)).squeeze(1)
        if self.nout == 1:
            out = c.view(-1)
        elif self.uncertainty and nHit is not None:
            log_var = self.uncertainty_head(torch.cat([c, nHit.unsqueeze(1) / self.npmts * 2.0], dim=1))
            out = (c, log_var)
        elif self.uncertainty2:
            log_var = self.uncertainty_head(c)
            out = (c, log_var)
        else:
            out = c

        if vae_aux is not None:
            return out, vae_aux
        return out
