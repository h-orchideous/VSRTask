from __future__ import annotations

import torch
from torch import nn


class DiffTransformer(nn.Module):
    """Predict the current-frame KV difference from a causal frame window.

    Input is ``[B, W, L, P, D]`` (or ``[W, L, P, D]`` without a batch),
    where W is the temporal window, L the Qwen layer count, P the visual
    token count, and D the KV feature dimension. The output contains only
    the prediction for the last/current frame.
    """

    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int | None = None,
        dropout: float = 0.0,
        max_window: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if max_window <= 0:
            raise ValueError("max_window must be positive")
        ff_dim = ff_dim or d_model * 4
        self.max_window = max_window
        self.input_proj = (
            nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_window, d_model))
        nn.init.normal_(self.temporal_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = (
            nn.Identity() if d_model == input_dim else nn.Linear(d_model, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.ndim == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,W,L,P,D] or [W,L,P,D], got {tuple(x.shape)}"
            )
        batch, window, layers, patches, dim = x.shape
        if window > self.max_window:
            raise ValueError(
                f"Window length {window} exceeds max_window={self.max_window}"
            )

        y = self.input_proj(x)
        y = y + self.temporal_pos[:, :window].view(1, window, 1, 1, -1)
        # Each Qwen layer and visual patch is one temporal sequence.
        y = y.permute(0, 2, 3, 1, 4).reshape(
            batch * layers * patches, window, -1
        )
        y = self.blocks(y)
        y = self.output_norm(y[:, -1])
        y = self.output_proj(y).reshape(batch, layers, patches, dim)
        return y.squeeze(0) if squeeze_batch else y


class KDiffTransformer(DiffTransformer):
    pass


class VDiffTransformer(DiffTransformer):
    pass
