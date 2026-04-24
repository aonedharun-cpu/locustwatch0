#!/usr/bin/env python3
"""
src/models/architecture.py
---------------------------
LocustNet: spatiotemporal neural model for desert locust outbreak prediction.

Architecture
------------
  Temporal branch
    LSTM (2 layers, hidden=128)
    Multi-head self-attention (4 heads) with residual + LayerNorm
    -> context vector h  [B, hidden]

  Spatial branch (GNN-style neighbourhood aggregation)
    Linear -> ReLU -> Linear on the last time step's nbr features
    -> spatial vector s  [B, gnn_hidden]

  Fusion
    Concat(h, s) -> Linear -> ReLU -> Dropout
    -> fused  [B, hidden]

  Output heads
    binary_head  : Linear -> logit          [B]   (outbreak_30d)
    phase_head   : Linear -> logits         [B,4] (phase_class 0-3)
    uncertainty  : Linear -> sigmoid        [B]   (MC Dropout variance proxy)

Uncertainty estimation
----------------------
  At inference time, call model.mc_predict(x, n_samples=30) to get
  mean probability + std across MC Dropout forward passes.

Input convention
----------------
  x  : (B, seq_len, n_features)
  The LAST n_nbr_features columns of x are neighbour (spatial) features.
  n_self_features and n_nbr_features are set at construction from the dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocustNet(nn.Module):
    def __init__(
        self,
        n_self_features: int,
        n_nbr_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        gnn_hidden: int = 64,
        dropout: float = 0.3,
        n_phases: int = 4,
    ):
        super().__init__()
        self.n_self = n_self_features
        self.n_nbr  = n_nbr_features
        self.hidden_size = hidden_size

        # ── Temporal branch ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=n_self_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Multi-head self-attention over LSTM output sequence
        self.attn      = nn.MultiheadAttention(hidden_size, num_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_drop = nn.Dropout(dropout)

        # ── Spatial branch (GNN-style) ────────────────────────────────────────
        self.nbr_encoder = nn.Sequential(
            nn.Linear(n_nbr_features, gnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + gnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        self.binary_head  = nn.Linear(hidden_size, 1)
        self.phase_head   = nn.Linear(hidden_size, n_phases)

        # MC Dropout uncertainty head
        self.mc_drop      = nn.Dropout(dropout)
        self.uncert_head  = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() == 2:
                nn.init.xavier_uniform_(p)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, seq_len, n_self + n_nbr)
        Returns
        -------
        binary_logit : (B,)
        phase_logit  : (B, 4)
        uncertainty  : (B,)  -- sigmoid output in [0,1]
        """
        x_self = x[:, :, :self.n_self]       # (B, seq_len, n_self)
        x_nbr  = x[:, -1, self.n_self:]      # (B, n_nbr)  last time step only

        # LSTM
        lstm_out, _ = self.lstm(x_self)       # (B, seq_len, hidden)

        # Multi-head attention with residual
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        h = self.attn_norm(lstm_out + self.attn_drop(attn_out))
        h = h[:, -1, :]                       # take last step (B, hidden)

        # Spatial branch
        s = self.nbr_encoder(x_nbr)           # (B, gnn_hidden)

        # Fusion
        fused = self.fusion(torch.cat([h, s], dim=1))   # (B, hidden)

        binary_logit = self.binary_head(fused).squeeze(1)    # (B,)
        phase_logit  = self.phase_head(fused)                 # (B, 4)
        uncertainty  = torch.sigmoid(
            self.uncert_head(self.mc_drop(fused))
        ).squeeze(1)                                          # (B,)

        return binary_logit, phase_logit, uncertainty

    # ── MC Dropout inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, n_samples: int = 30):
        """
        Monte Carlo Dropout inference.
        Keeps dropout active across n_samples forward passes.

        Returns
        -------
        mean_prob  : (B,)  mean outbreak probability
        std_prob   : (B,)  epistemic uncertainty (std of probabilities)
        """
        self.train()   # enable dropout
        probs = []
        for _ in range(n_samples):
            logit, _, _ = self.forward(x)
            probs.append(torch.sigmoid(logit))
        self.eval()

        probs = torch.stack(probs, dim=0)   # (n_samples, B)
        return probs.mean(0), probs.std(0)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(n_self_features: int, n_nbr_features: int,
                cfg: dict | None = None) -> LocustNet:
    """Build LocustNet from config dict (or use defaults)."""
    cfg = cfg or {}
    lstm_cfg  = cfg.get("lstm",      {})
    attn_cfg  = cfg.get("attention", {})
    gnn_cfg   = cfg.get("gnn",       {})

    return LocustNet(
        n_self_features=n_self_features,
        n_nbr_features=n_nbr_features,
        hidden_size=lstm_cfg.get("hidden_size", 128),
        num_layers=lstm_cfg.get("num_layers",   2),
        dropout=lstm_cfg.get("dropout",         0.3),
        num_heads=attn_cfg.get("num_heads",     4),
        gnn_hidden=gnn_cfg.get("hidden_size",   64),
    )
