#!/usr/bin/env python3
"""Compute VSR K/V deltas with Qwen3.5 full-attention layers.

Qwen3.5-9B is a hybrid model. Its 24 linear-attention layers do not expose
the same K/V representation as standard attention. This entry point captures
only the eight full-attention ``self_attn.k_proj/v_proj`` layers (3, 7, ...,
31), which have matching K and V dimensions and can be consumed by the
existing temporal PatchDiff detector.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError


LEGACY_TOOL = Path("/data1/ZhangHuayu/tools/compute_vsr_kv_diff.py")
DEFAULT_MODEL = Path("/data1/ZhangHuayu/models/Qwen3.5-9B")


def load_base_tool():
    spec = importlib.util.spec_from_file_location("vsr_compute_kv_diff_base", LEGACY_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base K/V diff tool: {LEGACY_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_qwen35(model_path: Path, device: str, dtype_name: str):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    full_layers = [
        name.rsplit(".self_attn.", 1)[0]
        for name, _ in model.named_modules()
        if name.endswith("self_attn.k_proj")
    ]
    if len(full_layers) != 8:
        raise RuntimeError(
            f"Expected 8 Qwen3.5 full-attention layers, found {len(full_layers)}: "
            f"{full_layers}"
        )
    print(
        "[qwen35-kv] capturing 8 full-attention layers; "
        "linear-attention recurrent states are intentionally excluded",
        flush=True,
    )
    return model, processor


def argument_path(name: str, default: Path) -> Path:
    if name in sys.argv:
        return Path(sys.argv[sys.argv.index(name) + 1])
    return default


def argument_values(name: str) -> list[str] | None:
    if name not in sys.argv:
        return None
    start = sys.argv.index(name) + 1
    values: list[str] = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def recover_missing_clean_frames(base) -> int:
    """Materialize only the four clean GT frames needed by each sample."""
    from train_patch_detector import ensure_clean_frame, resolve_clean_video

    manifest = argument_path(
        "--manifest", Path("/data1/ZhangHuayu/datasets/VSR-Train/manifest.json")
    )
    frames_root = argument_path(
        "--frames-root", Path("/data1/ZhangHuayu/tmp/vsr_train_frames")
    )
    samples = base.load_samples(manifest)

    sample_range = argument_values("--sample-range")
    sample_ids = argument_values("--sample-ids")
    if sample_range:
        if len(sample_range) != 2:
            raise ValueError("--sample-range requires START END")
        start, end = map(int, sample_range)
        wanted = {f"{number:08d}" for number in range(start, end + 1)}
    elif sample_ids:
        wanted = {base.normalize_id(value) for value in sample_ids}
    else:
        wanted = None

    recovered = 0
    for sample in samples:
        sample_id = base.normalize_id(sample["sample_id"])
        if wanted is not None and sample_id not in wanted:
            continue
        source_value = sample.get("source_video")
        if not source_value:
            raise KeyError(f"Sample {sample_id} has no source_video in {manifest}")
        source_video = Path(str(source_value))
        edited_video_value = sample.get("video_path")
        edited_video = None
        if edited_video_value:
            edited_video = Path(str(edited_video_value))
            if not edited_video.is_absolute():
                edited_video = manifest.parent / edited_video
        clean_video = resolve_clean_video(source_video, edited_video)

        for event in sample.get("edit_specs", []):
            frame_idx = int(event["frame_idx"])
            clean_path = frames_root / sample_id / f"{frame_idx + 1:06d}.png"
            if valid_image(clean_path):
                continue
            if clean_path.exists():
                clean_path.unlink()
            ensure_clean_frame(clean_video, frame_idx, clean_path)
            if not valid_image(clean_path):
                raise RuntimeError(f"Recovered clean frame is invalid: {clean_path}")
            recovered += 1
            print(
                f"[recover-clean] sample={sample_id} frame_idx={frame_idx} "
                f"source={clean_video}",
                flush=True,
            )
    print(f"[recover-clean] completed recovered={recovered}", flush=True)
    return recovered


def write_provenance(output_root: Path, model_path: Path) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone_name": "Qwen3.5-9B",
        "backbone_model": str(model_path.resolve()),
        "captured_representation": "full_attention_k_proj_v_proj",
        "decoder_layer_indices": [3, 7, 11, 15, 19, 23, 27, 31],
        "num_qwen_layers": 8,
        "expected_num_visual_tokens_for_640x480": 300,
        "expected_kv_dim": 1024,
        "excluded_representation": "linear_attention_gated_delta_net_state",
    }
    temporary = output_root / f".qwen35_kv_metadata.tmp-{os.getpid()}.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_root / "qwen35_kv_metadata.json")


def main() -> int:
    base = load_base_tool()
    base.DEFAULT_MODEL = DEFAULT_MODEL
    base.load_model_and_processor = load_qwen35

    # The base parser reads its own DEFAULT_MODEL as a function default, so
    # make the Qwen3.5 path explicit unless the caller supplied --model.
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", str(DEFAULT_MODEL)])
    recover_missing_clean_frames(base)
    result = base.main()

    output_root = argument_path(
        "--output-root",
        Path("/data1/ZhangHuayu/datasets/VSR-Train/kv_diff_qwen35"),
    )
    write_provenance(output_root, argument_path("--model", DEFAULT_MODEL))
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
