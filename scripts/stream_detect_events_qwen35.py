#!/usr/bin/env python3
"""Qwen3.5 PatchDiff producer for the streaming event-memory pipeline.

This is deliberately a separate entry point from the Qwen2.5 producer. The
checkpoint and runtime feature backbone must both be Qwen3.5-9B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

import stream_detect_events as base


def checkpoint_path() -> Path:
    if "--checkpoint" not in sys.argv:
        raise ValueError("--checkpoint is required")
    return Path(sys.argv[sys.argv.index("--checkpoint") + 1])


def validate_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("model_summary", {}).get("backbone", {})
    name = str(backbone.get("name", ""))
    if "Qwen3.5" not in name:
        raise ValueError(
            f"Qwen3.5 stream producer requires a Qwen3.5 PatchDiff checkpoint; "
            f"checkpoint backbone is {name or 'unknown'}"
        )


def main() -> int:
    validate_checkpoint(checkpoint_path())
    base.Qwen2_5_VLForConditionalGeneration = Qwen3_5ForConditionalGeneration
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
