#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path
from datetime import timedelta
from statistics import mean
from typing import Any

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_vsi_predicted_mask_videos import score_video_frames
from visualize_patch_diff_masks import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether PatchDiffDetector retrieves the four edited "
            "frames from every complete VSR video. No visual enhancement is applied."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        # Video lengths vary substantially and some ranks may finish much
        # earlier than others. Keep the final synchronization from timing out
        # while the longest rank is still scanning its videos.
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    return device, rank, world_size


def load_ground_truth(path: Path) -> list[int]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    indices = sorted(int(entry["frame_idx"]) for entry in record["entries"])
    if len(indices) != 4:
        raise ValueError(f"Expected four edited frames in {path}, got {indices}")
    return indices


def tolerant_matches(predicted: list[int], target: list[int], tolerance: int) -> int:
    matched_prediction: dict[int, int] = {}

    def assign(truth: int, visited: set[int]) -> bool:
        candidates = sorted(
            (
                pred for pred in predicted
                if abs(pred - truth) <= tolerance and pred not in visited
            ),
            key=lambda pred: abs(pred - truth),
        )
        for pred in candidates:
            visited.add(pred)
            previous_truth = matched_prediction.get(pred)
            if previous_truth is None or assign(previous_truth, visited):
                matched_prediction[pred] = truth
                return True
        return False

    return sum(assign(truth, set()) for truth in target)


def average_precision(frame_rows: list[dict[str, Any]], target: set[int]) -> float:
    ranked = sorted(frame_rows, key=lambda row: row["event_score"], reverse=True)
    hits = 0
    precisions = []
    for rank, row in enumerate(ranked, start=1):
        if row["frame_idx"] in target:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(target)


def evaluate_one(
    sample_id: int,
    video_path: Path,
    truth: list[int],
    frame_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(frame_rows, key=lambda row: row["event_score"], reverse=True)
    predicted = sorted(int(row["frame_idx"]) for row in ranked[:4])
    truth_set = set(truth)
    score_by_frame = {int(row["frame_idx"]): float(row["event_score"]) for row in frame_rows}
    ranks = {
        int(row["frame_idx"]): rank
        for rank, row in enumerate(ranked, start=1)
    }
    clean_scores = [
        float(row["event_score"])
        for row in frame_rows
        if int(row["frame_idx"]) not in truth_set
    ]
    exact = tolerant_matches(predicted, truth, 0)
    return {
        "sample_id": sample_id,
        "video_path": str(video_path),
        "frame_count": len(frame_rows),
        "target_frames": truth,
        "predicted_top4_frames": predicted,
        "exact_matches": exact,
        "matches_tolerance_1": tolerant_matches(predicted, truth, 1),
        "matches_tolerance_3": tolerant_matches(predicted, truth, 3),
        "all_four_exact": exact == 4,
        "target_frame_scores": [score_by_frame[index] for index in truth],
        "target_frame_ranks": [ranks[index] for index in truth],
        "predicted_top4_scores": [score_by_frame[index] for index in predicted],
        "target_score_mean": mean(score_by_frame[index] for index in truth),
        "clean_score_mean": mean(clean_scores),
        "frame_average_precision": average_precision(frame_rows, truth_set),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    events = count * 4
    return {
        "samples": count,
        "frames": sum(row["frame_count"] for row in rows),
        "edited_events": events,
        "top4_exact_matches": sum(row["exact_matches"] for row in rows),
        "top4_exact_recall": sum(row["exact_matches"] for row in rows) / events,
        "top4_recall_tolerance_1": sum(row["matches_tolerance_1"] for row in rows) / events,
        "top4_recall_tolerance_3": sum(row["matches_tolerance_3"] for row in rows) / events,
        "videos_all_four_exact": sum(row["all_four_exact"] for row in rows),
        "videos_all_four_exact_rate": sum(row["all_four_exact"] for row in rows) / count,
        "mean_frame_average_precision": mean(row["frame_average_precision"] for row in rows),
        "mean_target_frame_score": mean(row["target_score_mean"] for row in rows),
        "mean_clean_frame_score": mean(row["clean_score_mean"] for row in rows),
    }


def main() -> int:
    args = parse_args()
    start, end = args.sample_range
    if start > end:
        raise ValueError("Invalid sample range")
    if not 0 < args.top_fraction <= 1:
        raise ValueError("top-fraction must be in (0, 1]")

    device, rank, world_size = setup_distributed()
    qwen_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    table = pd.read_parquet(args.source_parquet)
    table = table[
        table["video_path"].map(lambda value: start <= int(Path(value).stem) <= end)
    ].sort_values("video_path")
    expected = end - start + 1
    if len(table) != expected:
        raise ValueError(f"Expected {expected} samples, found {len(table)}")
    records = table.to_dict("records")
    local_records = records[rank::world_size]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

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

    # score_video_frames only reads these inference fields.
    args.events_per_video = 4
    args.min_event_gap = 0
    print(
        f"[frame-eval] rank={rank}/{world_size} samples={len(local_records)}",
        flush=True,
    )
    for position, record in enumerate(local_records, start=1):
        sample_id = int(Path(record["video_path"]).stem)
        output_path = sample_dir / f"{sample_id:08d}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] rank={rank} sample={sample_id:08d}", flush=True)
            continue
        video_path = args.dataset_root / record["video_path"]
        truth_path = args.kv_diff_root / f"{sample_id:08d}.pt"
        truth = load_ground_truth(truth_path)
        print(
            f"[sample] rank={rank} {position}/{len(local_records)} "
            f"id={sample_id:08d} truth={truth}",
            flush=True,
        )
        frame_rows = score_video_frames(
            video_path, qwen, processor, detector, device, qwen_dtype, args
        )
        row = evaluate_one(sample_id, video_path, truth, frame_rows)
        output_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(
            f"[result] id={sample_id:08d} pred={row['predicted_top4_frames']} "
            f"exact={row['exact_matches']}/4",
            flush=True,
        )

    if world_size > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(sample_dir.glob("*.json"))
            if start <= int(path.stem) <= end
        ]
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} reports, found {len(rows)}")
        result = {
            "checkpoint": str(args.checkpoint),
            "sample_range": [start, end],
            "selection": "global_top4_without_temporal_suppression",
            "top_fraction_for_frame_score": args.top_fraction,
            "metrics": aggregate(rows),
            "samples": rows,
        }
        output_path = args.output_dir / "frame_detection_results.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["metrics"], indent=2), flush=True)
        print(f"[saved] {output_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
