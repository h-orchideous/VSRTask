#!/usr/bin/env python3
"""Consume streaming event jobs with the Qwen3.5 semantic writer."""
from __future__ import annotations

from transformers import Qwen3_5ForConditionalGeneration

from caption_event_memory_qwen35 import describe_event
import stream_caption_events_qwen25 as base


def main() -> int:
    base.Qwen2_5_VLForConditionalGeneration = Qwen3_5ForConditionalGeneration
    base.describe_event = describe_event
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
