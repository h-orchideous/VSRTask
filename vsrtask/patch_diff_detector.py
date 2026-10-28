from __future__ import annotations

import torch
from torch import nn


class TemporalPatchDiffDetector(nn.Module):
    """Predict current-frame patch diff magnitudes from a causal K/V window.

    Inputs:
        k, v: [batch, window, qwen_layers, patches, kv_dim]
    Output:
        patch_diff: [batch, qwen_layers, patches]

    Each (Qwen layer, spatial patch) pair is treated as an independent
    temporal sequence. There is no attention across layers or patches.
    """

    def __init__(
        self,
        num_qwen_layers: int = 28,
        kv_dim: int = 512,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        max_window: int = 8,
        max_visual_tokens: int = 391,
        dropout: float = 0.0,
        attention_batch_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if max_window <= 0 or max_visual_tokens <= 0:
            raise ValueError("max_window and max_visual_tokens must be positive")
        if attention_batch_chunk_size <= 0:
            raise ValueError("attention_batch_chunk_size must be positive")

        self.num_qwen_layers = num_qwen_layers
        self.kv_dim = kv_dim
        self.max_window = max_window
        self.max_visual_tokens = max_visual_tokens
        self.attention_batch_chunk_size = attention_batch_chunk_size

        self.temporal_pos = nn.Parameter(torch.zeros(1, max_window, d_model))
        self.layer_pos = nn.Parameter(torch.zeros(1, num_qwen_layers, d_model))
        self.patch_pos = nn.Parameter(torch.zeros(1, max_visual_tokens, d_model))
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.layer_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)

        self.input_proj = nn.Linear(kv_dim * 2, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(block, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.diff_head = nn.Linear(d_model, 1)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if k.shape != v.shape or k.ndim != 5:
            raise ValueError(
                f"Expected matching K/V [B,W,L,P,D], got {k.shape} and {v.shape}"
            )
        batch, window, layers, patches, dim = k.shape
        if layers != self.num_qwen_layers or dim != self.kv_dim:
            raise ValueError(f"Unexpected K/V feature shape: {tuple(k.shape)}")
        if window > self.max_window or patches > self.max_visual_tokens:
            raise ValueError(f"Input exceeds configured limits: {tuple(k.shape)}")

        x = self.input_proj(torch.cat((k, v), dim=-1))
        x = x + self.temporal_pos[:, :window].view(1, window, 1, 1, -1)
        x = x + self.layer_pos[:, :layers].view(1, 1, layers, 1, -1)
        x = x + self.patch_pos[:, :patches].view(1, 1, 1, patches, -1)

        x = x.permute(0, 2, 3, 1, 4).reshape(
            batch * layers * patches, window, -1
        )
        causal_mask = torch.triu(
            torch.ones(window, window, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        chunks = []
        for start in range(0, x.shape[0], self.attention_batch_chunk_size):
            chunks.append(
                self.temporal_blocks(
                    x[start : start + self.attention_batch_chunk_size],
                    mask=causal_mask,
                )
            )
        current = self.output_norm(torch.cat(chunks, dim=0)[:, -1])
        return self.diff_head(current).view(batch, layers, patches)


def patch_diff_target(delta_k: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
    """Convert full K/V deltas into one RMS magnitude per layer and patch."""
    if delta_k.shape != delta_v.shape or delta_k.ndim not in (3, 4):
        raise ValueError(
            "Expected matching [L,P,D] or [B,L,P,D] delta K/V tensors"
        )
    return torch.sqrt(
        0.5
        * (
            delta_k.float().square().mean(dim=-1)
            + delta_v.float().square().mean(dim=-1)
        )
    )


def fuse_qwen_layers(
    layer_scores: torch.Tensor,
    mode: str = "mean",
    last_n_layers: int | None = None,
) -> torch.Tensor:
    """Fuse [L,P] or [B,L,P] scores into a spatial [P] or [B,P] map."""
    if layer_scores.ndim not in (2, 3):
        raise ValueError(f"Expected [L,P] or [B,L,P], got {layer_scores.shape}")
    if last_n_layers is not None:
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        layer_scores = layer_scores[..., -last_n_layers:, :]
    if mode == "mean":
        return layer_scores.mean(dim=-2)
    if mode == "max":
        return layer_scores.max(dim=-2).values
    raise ValueError(f"Unsupported layer fusion mode: {mode}")

