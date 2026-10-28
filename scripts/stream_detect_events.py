#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import site
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import av
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_masked_vsr_videos import overlay_top_tokens
from build_vsi_predicted_mask_videos import frame_anomaly_score, setup_distributed
from visualize_patch_diff_masks import build_detector
from vsrtask.patch_diff_detector import fuse_qwen_layers
from vsrtask.qwen_kv import capture_edited_kv_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream threshold-crossing frames into a semantic event queue."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.30)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sample_number(video_path: str) -> int:
    return int(Path(video_path).stem)


def event_file(relative_video: Path, frame_idx: int) -> Path:
    return relative_video.parent / relative_video.stem / f"frame_{frame_idx:06d}.json"


def queue_name(relative_video: Path, frame_idx: int) -> str:
    prefix = "__".join((*relative_video.parent.parts, relative_video.stem))
    return f"{prefix}__frame_{frame_idx:06d}.json"


def empty_episodic_memory(frame_idx: int, seconds: float) -> dict[str, Any]:
    return {
        "when": {"frame": frame_idx, "seconds": seconds},
        "where": None,
        "what": None,
        "narrative": None,
        "entities": [],
        "dynamic_scene_relationships": {
            "object_counts": [],
            "location_relationships": [],
        },
        "raw_response": None,
    }


def process_video(
    relative_video: Path,
    video_path: Path,
    qwen: torch.nn.Module,
    processor: Any,
    detector: torch.nn.Module,
    device: torch.device,
    qwen_dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    detection_root = args.run_root / "detection"
    queue_root = args.run_root / "queue"
    evidence_dir = detection_root / "evidence" / relative_video.parent / relative_video.stem
    score_path = detection_root / "frame_scores" / relative_video.parent / f"{relative_video.stem}.jsonl.gz"
    memory_path = detection_root / "memories" / relative_video.parent / f"{relative_video.stem}.json"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)

    window_k: deque[torch.Tensor] = deque(maxlen=detector.max_window)
    window_v: deque[torch.Tensor] = deque(maxlen=detector.max_window)
    detector_dtype = next(detector.parameters()).dtype
    events: list[dict[str, Any]] = []
    frame_count = 0
    started_at = time.perf_counter()

    with (
        av.open(str(video_path), mode="r") as source,
        gzip.open(score_path, "wt", encoding="utf-8") as score_file,
        torch.inference_mode(),
    ):
        stream = source.streams.video[0]
        frame_rate = float(stream.average_rate or stream.base_rate or 1.0)
        for frame_idx, frame in enumerate(source.decode(stream)):
            image = frame.to_image().convert("RGB")
            frame_k, frame_v = capture_edited_kv_image(
                qwen, processor, image, device, qwen_dtype
            )
            frame_k = frame_k.to(detector_dtype)
            frame_v = frame_v.to(detector_dtype)
            if not window_k:
                for _ in range(detector.max_window - 1):
                    window_k.append(frame_k)
                    window_v.append(frame_v)
            window_k.append(frame_k)
            window_v.append(frame_v)

            k = torch.stack(tuple(window_k)).unsqueeze(0).to(device)
            v = torch.stack(tuple(window_v)).unsqueeze(0).to(device)
            predicted_layers = detector(k, v)[0].float().clamp_min(0).cpu()
            patch_scores = fuse_qwen_layers(
                predicted_layers,
                mode=args.layer_fusion,
                last_n_layers=args.last_n_layers,
            )
            score = frame_anomaly_score(patch_scores, args.top_fraction)
            score_file.write(
                json.dumps({"frame_idx": frame_idx, "event_score": score}) + "\n"
            )
            frame_count = frame_idx + 1

            if score >= args.threshold:
                seconds = frame_idx / frame_rate
                overlay, patch_indices, grid = overlay_top_tokens(
                    image.copy(),
                    patch_scores,
                    args.top_fraction,
                    args.mask_alpha,
                    args.outline_width,
                )
                original_path = evidence_dir / f"frame_{frame_idx:06d}_original.jpg"
                overlay_path = evidence_dir / f"frame_{frame_idx:06d}_overlay.jpg"
                image.save(original_path, quality=95)
                overlay.save(overlay_path, quality=95)
                event = {
                    "event_id": len(events) + 1,
                    "temporal_order": len(events) + 1,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "peak_frame": frame_idx,
                    "start_seconds": seconds,
                    "end_seconds": seconds,
                    "peak_seconds": seconds,
                    "threshold_crossing_frames": 1,
                    "peak_score": score,
                    "score_margin": score - args.threshold,
                    "episodic_memory": empty_episodic_memory(frame_idx, seconds),
                    "evidence": {
                        "original_image": str(original_path),
                        "overlay_image": str(overlay_path),
                        "selected_patch_indices": patch_indices,
                        "token_grid": list(grid),
                    },
                }
                events.append(event)
                semantic_relative = event_file(relative_video, frame_idx)
                job = {
                    "schema_version": 1,
                    "job_type": "qwen25_event_caption",
                    "video_path": str(relative_video),
                    "semantic_event_path": str(semantic_relative),
                    "event": event,
                }
                atomic_json(queue_root / "pending" / queue_name(relative_video, frame_idx), job)
                print(
                    f"[enqueue] video={relative_video} frame={frame_idx} "
                    f"score={score:.6f} event={len(events)}",
                    flush=True,
                )

            if frame_count % 10 == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[detect] video={relative_video.name} frames={frame_count} "
                    f"latest_score={score:.6f} events={len(events)} "
                    f"fps={frame_count / max(elapsed, 1e-6):.3f}",
                    flush=True,
                )

    memory = {
        "schema_version": 1,
        "memory_type": "streaming_threshold_event_memory",
        "video_path": str(relative_video),
        "source_video": str(video_path),
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "detector": {
            "checkpoint": str(args.checkpoint),
            "threshold": args.threshold,
            "top_fraction": args.top_fraction,
            "layer_fusion": args.layer_fusion,
            "last_n_layers": args.last_n_layers,
            "merge_gap": -1,
            "max_memory_events": None,
        },
        "frame_scores": str(score_path),
        "event_count": len(events),
        "events": events,
        "dynamic_scene_relationships": {
            "object_count_evolution": [],
            "location_relationship_evolution": [],
            "count_source": None,
            "cross_event_object_tracking": False,
        },
    }
    atomic_json(memory_path, memory)
    return memory, len(events)


def main() -> int:
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    device, rank, world_size = setup_distributed()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    table = pd.read_parquet(args.source_parquet).sort_values("video_path")
    if args.sample_range:
        start, end = args.sample_range
        table = table[
            table["video_path"].map(
                lambda value: start <= sample_number(value) <= end
            )
        ]
    rows = table.to_dict("records")[rank::world_size]
    for name in ("pending", "processing", "done", "producers"):
        (args.run_root / "queue" / name).mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    detector = build_detector(checkpoint, device)

    videos = events = 0
    for position, row in enumerate(rows, start=1):
        relative_video = Path(row["video_path"])
        memory_path = (
            args.run_root / "detection" / "memories" /
            relative_video.parent / f"{relative_video.stem}.json"
        )
        if memory_path.is_file() and not args.overwrite:
            print(f"[skip] rank={rank} {relative_video}", flush=True)
            continue
        print(
            f"[video] rank={rank}/{world_size} {position}/{len(rows)} {relative_video}",
            flush=True,
        )
        _, event_count = process_video(
            relative_video,
            args.dataset_root / relative_video,
            qwen,
            processor,
            detector,
            device,
            dtype,
            args,
        )
        videos += 1
        events += event_count

    done = args.run_root / "queue" / "producers" / f"rank_{rank:03d}.done.json"
    atomic_json(done, {"rank": rank, "world_size": world_size, "videos": videos, "events": events})
    print(f"[producer-done] rank={rank} videos={videos} events={events}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
