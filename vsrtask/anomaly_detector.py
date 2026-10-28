from __future__ import annotations

import torch
from torch import nn

from .patch_diff_detector import (
    TemporalPatchDiffDetector as LegacyTemporalPatchDiffDetector,
)


class _TemporalDiffBranch(nn.Module):
    """Predict a K-only or V-only delta tensor for the current frame."""
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

        self.input_proj = nn.Linear(kv_dim, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_window, d_model))
        self.layer_pos = nn.Parameter(torch.zeros(1, num_qwen_layers, d_model))
        self.patch_pos = nn.Parameter(torch.zeros(1, max_visual_tokens, d_model))
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.layer_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)

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
        # Predict the complete KV feature delta, not only its magnitude.
        self.diff_head = nn.Linear(d_model, kv_dim)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,W,L,P,D], got {tuple(x.shape)}")
        _b, _w, layers, patches, _dim = x.shape
        if layers != self.num_qwen_layers:
            raise ValueError(
                f"Expected {self.num_qwen_layers} Qwen layers, got {layers}"
            )
        if patches > self.max_visual_tokens:
            raise ValueError(
                f"Expected at most {self.max_visual_tokens} patches, got {patches}"
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        b, w, layers, patches, _dim = x.shape
        if w > self.max_window:
            raise ValueError(f"Window length {w} exceeds max_window={self.max_window}")

        x = self.input_proj(x)
        x = x + self.temporal_pos[:, :w].view(1, w, 1, 1, -1)
        x = x + self.layer_pos[:, :layers].view(1, 1, layers, 1, -1)
        x = x + self.patch_pos[:, :patches].view(1, 1, 1, patches, -1)

        # Each (Qwen layer, patch) pair is a temporal sequence. Only the last
        # timestep is supervised.
        x = x.permute(0, 2, 3, 1, 4).reshape(b * layers * patches, w, -1)
        causal_mask = torch.triu(
            torch.ones(w, w, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        # A batch of B*L*P temporal streams can be very large even though the
        # temporal sequence itself is short. Chunking avoids CUDA attention
        # launch limits while preserving the exact same computation.
        chunks = []
        for start in range(0, x.shape[0], self.attention_batch_chunk_size):
            chunks.append(
                self.temporal_blocks(
                    x[start : start + self.attention_batch_chunk_size],
                    mask=causal_mask,
                )
            )
        x = torch.cat(chunks, dim=0)
        current = self.output_norm(x[:, -1])
        patch_delta = self.diff_head(current).view(b, layers, patches, self.kv_dim)
        return patch_delta


class TemporalPatchAnomalyDetector(nn.Module):
    """Detect edited visual tokens with independent K and V branches.

    Inputs are frozen-Qwen K/V features:
      k, v: [batch, window, qwen_layers, visual_tokens, kv_dim]

    Outputs keep K and V separate:
      predicted_k, predicted_v: [batch, qwen_layers, visual_tokens, kv_dim]
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
        self.num_qwen_layers = num_qwen_layers
        self.kv_dim = kv_dim
        self.max_window = max_window
        self.max_visual_tokens = max_visual_tokens
        branch_kwargs = dict(
            num_qwen_layers=num_qwen_layers,
            kv_dim=kv_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_window=max_window,
            max_visual_tokens=max_visual_tokens,
            dropout=dropout,
            attention_batch_chunk_size=attention_batch_chunk_size,
        )
        self.k_branch = _TemporalDiffBranch(**branch_kwargs)
        self.v_branch = _TemporalDiffBranch(**branch_kwargs)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        return self.k_branch(k), self.v_branch(v)


def diff_patch_magnitude(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
) -> torch.Tensor:
    """Return one continuous diff magnitude per layer and visual token.

    Accepts [L,P,D], [L,1,P,D], or batched [B,L,P,D] tensors and combines
    K/V RMS energy.
    """
    if delta_k.ndim == 4 and delta_k.shape[1] == 1:
        delta_k = delta_k.squeeze(1)
    if delta_v.ndim == 4 and delta_v.shape[1] == 1:
        delta_v = delta_v.squeeze(1)
    if delta_k.ndim not in (3, 4) or delta_v.ndim != delta_k.ndim:
        raise ValueError(
            "delta K/V must have shape [L,P,D], [L,1,P,D], or [B,L,P,D]"
        )
    if delta_k.shape != delta_v.shape:
        raise ValueError("delta K/V shapes must match")
    # Keep the Qwen-layer axis; reduce only the 512-dimensional KV feature.
    k_score = delta_k.float().pow(2).mean(dim=-1).sqrt()
    v_score = delta_v.float().pow(2).mean(dim=-1).sqrt()
    return torch.sqrt(0.5 * (k_score.square() + v_score.square()))


def diff_patch_components(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate K/V RMS diff targets, each shaped [L, P].

    This remains available for diagnostics and for converting a predicted
    full delta into a patch-level anomaly score.
    """
    if delta_k.ndim == 4:
        delta_k = delta_k.squeeze(1)
    if delta_v.ndim == 4:
        delta_v = delta_v.squeeze(1)
    if delta_k.ndim != 3 or delta_v.ndim != 3:
        raise ValueError("delta K/V must have shape [L,P,D] or [L,1,P,D]")
    if delta_k.shape != delta_v.shape:
        raise ValueError("delta K/V shapes must match")
    return (
        delta_k.float().pow(2).mean(dim=-1).sqrt(),
        delta_v.float().pow(2).mean(dim=-1).sqrt(),
    )


def fuse_kv_diff(
    predicted_k: torch.Tensor,
    predicted_v: torch.Tensor,
    mode: str = "max",
) -> torch.Tensor:
    """Fuse K/V scores only when a single patch score is needed."""
    if predicted_k.shape != predicted_v.shape:
        raise ValueError(
            f"Predicted K/V shape mismatch: {tuple(predicted_k.shape)} "
            f"vs {tuple(predicted_v.shape)}"
        )
    if mode == "max":
        return torch.maximum(predicted_k.float(), predicted_v.float())
    if mode == "mean":
        return 0.5 * (predicted_k.float() + predicted_v.float())
    if mode == "rms":
        return torch.sqrt(
            0.5 * (predicted_k.float().square() + predicted_v.float().square())
        )
    raise ValueError(f"Unsupported K/V fusion mode: {mode}")


def fuse_layer_diff(
    layer_diff: torch.Tensor,
    mode: str = "mean",
    last_n_layers: int | None = None,
) -> torch.Tensor:
    """Fuse [B,L,P] or [L,P] predictions into per-patch [B,P] / [P]."""
    if layer_diff.ndim not in (2, 3):
        raise ValueError(
            f"Expected [L,P] or [B,L,P], got {tuple(layer_diff.shape)}"
        )
    if last_n_layers is not None:
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        layer_diff = layer_diff[..., -last_n_layers:, :]
    if mode == "mean":
        return layer_diff.mean(dim=-2)
    if mode == "max":
        return layer_diff.max(dim=-2).values
    raise ValueError(f"Unsupported layer fusion mode: {mode}")


# Backward-compatible name for older local experiments.
diff_patch_score = diff_patch_magnitude


def top_quantile_patch_labels(
    score: torch.Tensor,
    positive_quantile: float = 0.9,
) -> torch.Tensor:
    """Convert a positive-frame score map into binary patch labels."""
    if not 0.0 < positive_quantile < 1.0:
        raise ValueError("positive_quantile must be in (0, 1)")
    threshold = torch.quantile(score.float(), positive_quantile)
    labels = score >= threshold
    # Ensure a non-empty positive target even when all scores are identical.
    if not labels.any():
        labels[score.argmax()] = True
    return labels.float()
