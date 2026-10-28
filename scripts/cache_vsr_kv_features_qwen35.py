#!/usr/bin/env python3
"""Cache Qwen3.5 full-attention K/V features for PatchDiff training."""
from __future__ import annotations

import sys

from transformers import Qwen3_5ForConditionalGeneration

import cache_vsr_kv_features as base


def main() -> int:
    # Reuse the established frame/window/recovery logic while replacing only
    # the frozen visual-language backbone class.
    base.Qwen2_5_VLForConditionalGeneration = Qwen3_5ForConditionalGeneration
    if "--backbone-name" not in sys.argv:
        sys.argv.extend(["--backbone-name", "Qwen3.5-9B-full-attention"])
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
