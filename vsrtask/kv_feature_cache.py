from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_sample_id(value: Any) -> str:
    text = str(value)
    return f"{int(text):08d}" if text.isdigit() else text


def feature_cache_key(path: Path) -> str:
    return str(path.resolve())


def load_feature_cache(
    cache_path: Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Memory-map a per-sample frozen-Qwen K/V feature cache."""
    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Invalid KV cache, missing frames list: {cache_path}")
    features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid frame item in KV cache: {cache_path}")
        path = Path(str(item["path"]))
        k = item["k"]
        v = item["v"]
        if not torch.is_tensor(k) or not torch.is_tensor(v) or k.shape != v.shape:
            raise ValueError(f"Invalid K/V tensors in KV cache: {cache_path}")
        features[feature_cache_key(path)] = (k, v)
    return features, payload

