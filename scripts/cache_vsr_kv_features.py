#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import site
import sys
from pathlib import Path

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path
        for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import torch
import torch.distributed as dist
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.detector_data import load_diff_records, make_windows_for_sample
from vsrtask.qwen_kv import capture_edited_kv
from train_patch_detector import (
    ensure_clean_frame,
    ensure_edited_frames,
    load_source_video_map,
    load_video_map,
    normalize_sample_id,
    probe_video_frame_count,
    resolve_clean_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-extract frozen Qwen K/V features for all unique frames used "
            "by the VSR patch-delta training windows."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--backbone-name",
        default="Qwen2.5-VL",
        help="Backbone identifier stored in the cache for compatibility checks.",
    )
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--negative-windows", type=int, default=4)
    parser.add_argument(
        "--cache-epochs",
        type=int,
        default=1,
        help=(
            "Number of training epochs whose negative-window seeds are included "
            "in the cache. Use the same value as the training --epochs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Storage dtype. float16 is recommended for the full K/V tensors.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_device(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed KV caching requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    return device, rank, world_size


def sample_id_from_path(path: Path) -> str:
    return normalize_sample_id(path.stem)


def cache_key(path: Path) -> str:
    return str(path.resolve())


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def ensure_valid_clean_frame(
    source_video: Path,
    frame_idx: int,
    frame_path: Path,
) -> None:
    if valid_image(frame_path):
        return
    if frame_path.exists():
        frame_path.unlink()
    ensure_clean_frame(source_video, frame_idx, frame_path)
    if not valid_image(frame_path):
        raise RuntimeError(
            f"Recovered frame is still invalid: sample frame_idx={frame_idx} "
            f"path={frame_path}"
        )


def main() -> int:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.negative_windows < 0:
        raise ValueError("--negative-windows must be non-negative")
    if args.cache_epochs <= 0:
        raise ValueError("--cache-epochs must be positive")
    if args.sample_range and args.sample_range[0] > args.sample_range[1]:
        raise ValueError("--sample-range START must be <= END")

    device, rank, world_size = setup_device(args)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    qwen_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    storage_dtype = qwen_dtype

    records = load_diff_records(
        args.kv_diff_root,
        tuple(args.sample_range) if args.sample_range else None,
        args.max_samples,
    )
    local_records = records[rank::world_size]
    video_map = load_video_map(args.manifest, args.dataset_root)
    source_video_map = load_source_video_map(args.manifest)

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        trust_remote_code=True,
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=qwen_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[cache_vsr_kv_features] rank={rank}/{world_size} "
        f"samples={len(local_records)} storage_dtype={args.dtype}",
        flush=True,
    )

    for record_path in local_records:
        sample_id = sample_id_from_path(record_path)
        output_path = args.output_root / f"{sample_id}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {sample_id} already exists: {output_path}", flush=True)
            continue

        record = torch.load(record_path, map_location="cpu", weights_only=False)
        recovered = ensure_edited_frames(record, video_map)
        source_video = source_video_map.get(sample_id)
        edited_video = video_map.get(sample_id)
        if source_video is None:
            raise KeyError(f"No source_video for sample {sample_id}")
        source_video = resolve_clean_video(source_video, edited_video)
        frame_count = probe_video_frame_count(str(source_video))

        unique_paths: dict[str, tuple[int, Path]] = {}
        for epoch in range(args.cache_epochs):
            windows = make_windows_for_sample(
                record,
                args.frames_root,
                args.window_size,
                args.negative_windows,
                args.seed + epoch,
                frame_count=frame_count,
            )
            for window in windows:
                for frame_idx, frame_path in zip(
                    window.frame_indices,
                    window.frame_paths,
                ):
                    if not valid_image(frame_path):
                        # Edited-frame paths should already be valid after
                        # ensure_edited_frames; clean paths can be recovered
                        # from the source video if a previous extraction left
                        # a zero-byte or corrupted PNG.
                        if Path(str(frame_path)).parent.name == sample_id:
                            ensure_valid_clean_frame(
                                source_video,
                                frame_idx,
                                frame_path,
                            )
                        elif not valid_image(frame_path):
                            raise RuntimeError(
                                f"Invalid edited frame cannot be recovered "
                                f"from the clean source video: {frame_path}"
                            )
                    unique_paths[cache_key(frame_path)] = (frame_idx, frame_path)

        frame_items = []
        for frame_number, (path_key, (frame_idx, frame_path)) in enumerate(
            sorted(unique_paths.items())
        ):
            try:
                frame_k, frame_v = capture_edited_kv(
                    qwen,
                    processor,
                    frame_path,
                    device,
                    qwen_dtype,
                )
            except UnidentifiedImageError:
                if frame_path.parent.name != sample_id:
                    raise
                ensure_valid_clean_frame(source_video, frame_idx, frame_path)
                frame_k, frame_v = capture_edited_kv(
                    qwen,
                    processor,
                    frame_path,
                    device,
                    qwen_dtype,
                )
            if frame_k.ndim != 3 or frame_v.ndim != 3:
                raise RuntimeError(
                    f"Unexpected K/V shape for {frame_path}: "
                    f"{tuple(frame_k.shape)} / {tuple(frame_v.shape)}"
                )
            if frame_k.shape != frame_v.shape:
                raise RuntimeError(
                    f"K/V shape mismatch for {frame_path}: "
                    f"{tuple(frame_k.shape)} / {tuple(frame_v.shape)}"
                )
            frame_items.append(
                {
                    "path": path_key,
                    "frame_idx": int(frame_idx),
                    "k": frame_k.to(storage_dtype),
                    "v": frame_v.to(storage_dtype),
                }
            )
            if (frame_number + 1) % 10 == 0 or frame_number + 1 == len(unique_paths):
                print(
                    f"[cache] sample={sample_id} "
                    f"frames={frame_number + 1}/{len(unique_paths)}",
                    flush=True,
                )

        if not frame_items:
            raise RuntimeError(f"No required frames found for sample {sample_id}")
        first_k = frame_items[0]["k"]
        payload = {
            "sample_id": sample_id,
            "backbone_name": args.backbone_name,
            "backbone_model": str(args.model.resolve()),
            "attention_layers": "self_attn_full_attention",
            "dtype": str(first_k.dtype).replace("torch.", ""),
            "num_frames": len(frame_items),
            "num_qwen_layers": int(first_k.shape[0]),
            "num_visual_tokens": int(first_k.shape[1]),
            "kv_dim": int(first_k.shape[2]),
            "cache_epochs": args.cache_epochs,
            "window_size": args.window_size,
            "negative_windows": args.negative_windows,
            "seed": args.seed,
            "frame_count": frame_count,
            "recovered_edited_frames": recovered,
            "frames": frame_items,
        }
        temporary = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}.pt"
        )
        torch.save(payload, temporary)
        temporary.replace(output_path)
        print(
            f"[saved] {output_path} frames={len(frame_items)} "
            f"shape={tuple(first_k.shape)}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier(device_ids=[device.index])
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
