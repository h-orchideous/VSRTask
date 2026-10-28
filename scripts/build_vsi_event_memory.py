#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import site
import sys
from pathlib import Path
from typing import Any

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path if Path(path).resolve() != Path(user_site).resolve()
    ]

import av
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_masked_vsr_videos import overlay_top_tokens
from build_vsi_predicted_mask_videos import score_video_frames, setup_distributed
from visualize_patch_diff_masks import build_detector
from vsrtask.event_memory import EventSpan, group_threshold_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write threshold-triggered PatchDiff events as explicit memory."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=-1,
        help=(
            "Disabled by default so every threshold-crossing frame remains a "
            "separate ordered event. Set to a nonnegative value only for an "
            "explicit temporal+patch-overlap deduplication experiment."
        ),
    )
    parser.add_argument("--min-patch-iou", type=float, default=0.10)
    parser.add_argument(
        "--max-memory-events",
        type=int,
        default=None,
        help=(
            "Optional resource cap. By default every threshold-crossing frame "
            "is retained as an independent event."
        ),
    )
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.3)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--caption-events", action="store_true")
    parser.add_argument("--caption-max-new-tokens", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate_image_text(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[:, batch["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def describe_event(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    frame_idx: int,
    timestamp: float,
) -> dict[str, Any]:
    narrative_prompt = (
        "Describe this unusual video event in at most 45 words. Use exactly: "
        "Where: <scene and spatial context>; What: <unusual entity and event>; "
        "Relations: <entity relative to nearby objects>. "
        f"This is frame {frame_idx} at {timestamp:.3f} seconds."
    )
    narrative = generate_image_text(
        model,
        processor,
        image,
        device,
        dtype,
        narrative_prompt,
        min(max_new_tokens, 96),
    )
    count_prompt = (
        "Count visible objects in this frame. The FIRST pair MUST be the unusual "
        "foreground entity described by this event: "
        f"{narrative} "
        "Then include up to seven nearby salient object types. Output only "
        "semicolon-separated name=count pairs, for example: dog=1; sofa=1; "
        "table=2. Use singular lowercase names and integer counts. Do not output "
        "explanations."
    )
    count_response = generate_image_text(
        model,
        processor,
        image,
        device,
        dtype,
        count_prompt,
        64,
    )
    counts = []
    for name, count in re.findall(
        r"([^;=:\n{}]+?)\s*[=:]\s*(\d+)", count_response
    ):
        normalized = name.strip().strip('"\'').lower()
        if normalized:
            counts.append({"name": normalized, "count": int(count)})

    fields = {}
    for key in ("Where", "What", "Relations"):
        match = re.search(
            rf"{key}:\s*(.*?)(?=;\s*(?:Where|What|Relations):|$)",
            narrative,
            re.IGNORECASE,
        )
        fields[key.lower()] = match.group(1).strip() if match else None
    relations = [fields["relations"]] if fields["relations"] else []
    return {
        "narrative": narrative,
        "where": fields["where"],
        "what": fields["what"],
        "object_counts": counts,
        "spatial_relations": relations,
        "raw_response": {
            "narrative": narrative,
            "counts": count_response,
        },
    }


def write_frame_scores(path: Path, frame_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(
                json.dumps(
                    {
                        "frame_idx": int(row["frame_idx"]),
                        "event_score": float(row["event_score"]),
                    }
                )
                + "\n"
            )


def write_event_evidence(
    video_path: Path,
    events: list[EventSpan],
    frame_rows: list[dict[str, Any]],
    evidence_dir: Path,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    event_by_peak = {event.peak_frame: event for event in events}
    row_by_frame = {int(row["frame_idx"]): row for row in frame_rows}
    records = []
    with av.open(str(video_path), mode="r") as source:
        stream = source.streams.video[0]
        rate_value = float(stream.average_rate or stream.base_rate or 1.0)
        for frame_idx, frame in enumerate(source.decode(stream)):
            event = event_by_peak.get(frame_idx)
            if event is None:
                continue
            image = frame.to_image().convert("RGB")
            row = row_by_frame[frame_idx]
            overlay, patch_indices, grid = overlay_top_tokens(
                image.copy(),
                row["patch_scores"],
                args.top_fraction,
                args.mask_alpha,
                args.outline_width,
            )
            original_path = evidence_dir / f"frame_{frame_idx:06d}_original.jpg"
            overlay_path = evidence_dir / f"frame_{frame_idx:06d}_overlay.jpg"
            image.save(original_path, quality=95)
            overlay.save(overlay_path, quality=95)
            episodic = None
            if args.caption_events:
                episodic = describe_event(
                    model,
                    processor,
                    image,
                    device,
                    dtype,
                    args.caption_max_new_tokens,
                    frame_idx,
                    frame_idx / rate_value,
                )
            records.append(
                {
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "peak_frame": event.peak_frame,
                    "start_seconds": event.start_frame / rate_value,
                    "end_seconds": event.end_frame / rate_value,
                    "peak_seconds": event.peak_frame / rate_value,
                    "threshold_crossing_frames": event.frame_count,
                    "peak_score": event.peak_score,
                    "score_margin": event.peak_score - args.threshold,
                    "episodic_memory": {
                        "when": {
                            "frame": frame_idx,
                            "seconds": frame_idx / rate_value,
                        },
                        "where": episodic.get("where") if episodic else None,
                        "what": episodic.get("what") if episodic else None,
                        "narrative": (
                            episodic.get("narrative") if episodic else None
                        ),
                        "entities": (
                            episodic.get("object_counts", []) if episodic else []
                        ),
                        "dynamic_scene_relationships": {
                            "object_counts": (
                                episodic.get("object_counts", []) if episodic else []
                            ),
                            "location_relationships": (
                                episodic.get("spatial_relations", [])
                                if episodic
                                else []
                            ),
                        },
                        "raw_response": (
                            episodic.get("raw_response") if episodic else None
                        ),
                    },
                    "evidence": {
                        "original_image": str(original_path),
                        "overlay_image": str(overlay_path),
                        "selected_patch_indices": patch_indices,
                        "token_grid": list(grid),
                    },
                }
            )
    records.sort(key=lambda record: record["peak_frame"])
    for event_id, record in enumerate(records, start=1):
        record["event_id"] = event_id
        record["temporal_order"] = event_id
    return records, rate_value


def main() -> int:
    args = parse_args()
    if args.max_memory_events is not None and args.max_memory_events <= 0:
        raise ValueError("--max-memory-events must be positive")
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")

    device, rank, world_size = setup_distributed()
    qwen_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    table = pd.read_parquet(args.source_parquet).sort_values("video_path")
    if args.sample_range:
        start, end = args.sample_range
        table = table[
            table["video_path"].map(
                lambda value: start <= int(Path(value).stem) <= end
            )
        ]
    rows = table.to_dict("records")
    local_rows = rows[rank::world_size]
    args.output_root.mkdir(parents=True, exist_ok=True)
    memory_root = args.output_root / "memories"
    evidence_root = args.output_root / "evidence"
    score_root = args.output_root / "frame_scores"

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=qwen_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    detector = build_detector(checkpoint, device)

    print(
        f"[event-memory] rank={rank}/{world_size} videos={len(local_rows)} "
        f"threshold={args.threshold} merge_gap={args.merge_gap}",
        flush=True,
    )
    for position, row in enumerate(local_rows, start=1):
        relative_path = Path(row["video_path"])
        memory_path = memory_root / relative_path.parent / f"{relative_path.stem}.json"
        if memory_path.exists() and not args.overwrite:
            print(f"[skip] rank={rank} {relative_path}", flush=True)
            continue
        video_path = args.dataset_root / relative_path
        print(
            f"[video] rank={rank} {position}/{len(local_rows)} {relative_path}",
            flush=True,
        )
        frame_rows = score_video_frames(
            video_path, qwen, processor, detector, device, qwen_dtype, args
        )
        events = group_threshold_events(
            frame_rows,
            threshold=args.threshold,
            merge_gap=args.merge_gap,
            top_fraction=args.top_fraction,
            min_patch_iou=args.min_patch_iou,
            max_events=args.max_memory_events,
        )
        score_path = score_root / relative_path.parent / f"{relative_path.stem}.jsonl.gz"
        write_frame_scores(score_path, frame_rows)
        records, frame_rate = write_event_evidence(
            video_path,
            events,
            frame_rows,
            evidence_root / relative_path.parent / relative_path.stem,
            qwen,
            processor,
            device,
            qwen_dtype,
            args,
        )
        memory = {
            "schema_version": 1,
            "memory_type": "threshold_triggered_temporal_event_memory",
            "video_path": str(relative_path),
            "source_video": str(video_path),
            "frame_rate": frame_rate,
            "frame_count": len(frame_rows),
            "detector": {
                "checkpoint": str(args.checkpoint),
                "threshold": args.threshold,
                "top_fraction": args.top_fraction,
                "layer_fusion": args.layer_fusion,
                "last_n_layers": args.last_n_layers,
                "merge_gap": args.merge_gap,
                "min_patch_iou": args.min_patch_iou,
                "max_memory_events": args.max_memory_events,
            },
            "frame_scores": str(score_path),
            "event_count": len(records),
            "events": records,
            "dynamic_scene_relationships": {
                "object_count_evolution": [
                    {
                        "event_id": record["event_id"],
                        "frame": record["peak_frame"],
                        "seconds": record["peak_seconds"],
                        "counts": record["episodic_memory"][
                            "dynamic_scene_relationships"
                        ]["object_counts"],
                    }
                    for record in records
                ],
                "location_relationship_evolution": [
                    {
                        "event_id": record["event_id"],
                        "frame": record["peak_frame"],
                        "seconds": record["peak_seconds"],
                        "relations": record["episodic_memory"][
                            "dynamic_scene_relationships"
                        ]["location_relationships"],
                    }
                    for record in records
                ],
                "count_source": "qwen2.5-vl_event_observation",
                "cross_event_object_tracking": False,
            },
        }
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
        print(
            f"[saved] {memory_path} events={len(records)} "
            f"peaks={[record['peak_frame'] for record in records]}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        memory_paths = sorted(memory_root.rglob("*.json"))
        summary = {
            "schema_version": 1,
            "videos": len(memory_paths),
            "threshold": args.threshold,
            "merge_gap": args.merge_gap,
            "min_patch_iou": args.min_patch_iou,
            "max_memory_events": args.max_memory_events,
            "caption_events": args.caption_events,
            "memories": [str(path) for path in memory_paths],
        }
        summary_path = args.output_root / "memory_manifest.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[complete] {summary_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
