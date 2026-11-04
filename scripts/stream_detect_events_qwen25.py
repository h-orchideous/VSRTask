#!/usr/bin/env python3
"""Qwen2.5 PatchDiff producer for the streaming event-memory pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import stream_detect_events as base


def checkpoint_path() -> Path:
    if "--checkpoint" not in sys.argv:
        raise ValueError("--checkpoint is required")
    return Path(sys.argv[sys.argv.index("--checkpoint") + 1])


def validate_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("model_summary", {}).get("backbone", {})
    name = str(backbone.get("name", ""))
    if "Qwen2.5" not in name:
        raise ValueError(
            "Qwen2.5 stream producer requires a Qwen2.5 PatchDiff checkpoint; "
            f"checkpoint backbone is {name or 'unknown'}"
        )


def main() -> int:
    validate_checkpoint(checkpoint_path())
    base.JOB_TYPE = "qwen25_event_caption"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
