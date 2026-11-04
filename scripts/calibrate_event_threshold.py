#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a streaming PatchDiff threshold from every labeled frame, "
            "including the complete clean-frame score distribution."
        )
    )
    parser.add_argument("--detection-results", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--max-clean-frame-fpr",
        type=float,
        help="Optional upper bound on the fraction of clean frames that trigger memory.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        help="Optional lower bound on frame-level precision during calibration.",
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        help="Optional JSON file containing the complete threshold trade-off curve.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def valid_probability(value: float | None, name: str) -> None:
    if value is not None and not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


def full_frame_labels(source: dict[str, Any]) -> tuple[list[tuple[float, bool, int]], list[int]]:
    rows: list[tuple[float, bool, int]] = []
    sample_ids: list[int] = []
    for sample in source.get("samples", []):
        sample_id = int(sample["sample_id"])
        targets = {int(frame_idx) for frame_idx in sample["target_frames"]}
        frame_scores = sample.get("frame_scores")
        if not isinstance(frame_scores, list):
            raise ValueError(
                "Full frame scores are missing. Re-run "
                "evaluate_temporal_frame_detection.py with --save-frame-scores."
            )
        seen: set[int] = set()
        for item in frame_scores:
            frame_idx = int(item["frame_idx"])
            score = float(item["event_score"])
            if not math.isfinite(score):
                raise ValueError(
                    f"Non-finite score for sample={sample_id:08d} frame={frame_idx}"
                )
            if frame_idx in seen:
                raise ValueError(
                    f"Duplicate score for sample={sample_id:08d} frame={frame_idx}"
                )
            seen.add(frame_idx)
            rows.append((score, frame_idx in targets, sample_id))
        missing_targets = sorted(targets - seen)
        if missing_targets:
            raise ValueError(
                f"Missing target scores for sample={sample_id:08d}: {missing_targets}"
            )
        sample_ids.append(sample_id)
    if not rows:
        raise ValueError("No full frame scores found")
    return rows, sorted(sample_ids)


def build_curve(rows: list[tuple[float, bool, int]]) -> tuple[list[dict[str, Any]], int, int]:
    total_positive = sum(is_positive for _, is_positive, _ in rows)
    total_clean = len(rows) - total_positive
    video_count = len({sample_id for _, _, sample_id in rows})
    if not total_positive or not total_clean:
        raise ValueError("Calibration requires both edited and clean frames")

    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    video_counts: dict[int, int] = defaultdict(int)
    selected_positive = 0
    selected_clean = 0
    cursor = 0
    curve = []
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            _, is_positive, sample_id = ordered[cursor]
            if is_positive:
                selected_positive += 1
            else:
                selected_clean += 1
            video_counts[sample_id] += 1
            cursor += 1
        selected = selected_positive + selected_clean
        curve.append(
            {
                "threshold": threshold,
                "recall": selected_positive / total_positive,
                "precision": selected_positive / selected if selected else 0.0,
                "clean_frame_fpr": selected_clean / total_clean,
                "true_positives": selected_positive,
                "false_positives": selected_clean,
                "mean_events_per_video": selected / video_count,
                "max_events_per_video": max(video_counts.values()),
            }
        )
    return curve, total_positive, total_clean


def main() -> int:
    args = parse_args()
    valid_probability(args.target_recall, "--target-recall")
    valid_probability(args.max_clean_frame_fpr, "--max-clean-frame-fpr")
    valid_probability(args.min_precision, "--min-precision")

    source = json.loads(args.detection_results.read_text(encoding="utf-8"))
    rows, sample_ids = full_frame_labels(source)
    curve, positive_count, clean_count = build_curve(rows)
    eligible = [row for row in curve if row["recall"] >= args.target_recall]
    if args.max_clean_frame_fpr is not None:
        eligible = [
            row
            for row in eligible
            if row["clean_frame_fpr"] <= args.max_clean_frame_fpr
        ]
    if args.min_precision is not None:
        eligible = [
            row for row in eligible if row["precision"] >= args.min_precision
        ]
    if not eligible:
        raise RuntimeError(
            "No threshold satisfies the requested calibration constraints. "
            "Lower --target-recall or relax the clean-FPR/precision limit."
        )

    # The highest threshold satisfying the constraints minimizes memory writes
    # while preserving the requested recall on the held-out calibration split.
    selected = max(eligible, key=lambda row: row["threshold"])
    result = {
        "schema_version": 2,
        "threshold": selected["threshold"],
        "selection_rule": "highest_threshold_meeting_constraints",
        "target_recall": args.target_recall,
        "max_clean_frame_fpr": args.max_clean_frame_fpr,
        "min_precision": args.min_precision,
        "measured_recall": selected["recall"],
        "measured_precision": selected["precision"],
        "measured_clean_frame_fpr": selected["clean_frame_fpr"],
        "true_positives": selected["true_positives"],
        "false_positives": selected["false_positives"],
        "positive_count": positive_count,
        "clean_frame_count": clean_count,
        "mean_threshold_events_per_video": selected["mean_events_per_video"],
        "max_threshold_events_per_video": selected["max_events_per_video"],
        "calibration_sample_ids": sample_ids,
        "calibration_source": str(args.detection_results),
        "calibration_scope": (
            "All scored edited and clean frames. No Top-K frame selection or "
            "known event-count cap is used during calibration."
        ),
        "score_definition": {
            "layer_fusion": source.get("layer_fusion", "mean"),
            "top_fraction": source.get("top_fraction_for_frame_score"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.curve_output:
        args.curve_output.parent.mkdir(parents=True, exist_ok=True)
        args.curve_output.write_text(
            json.dumps(
                {
                    "detection_results": str(args.detection_results),
                    "positive_count": positive_count,
                    "clean_frame_count": clean_count,
                    "curve": curve,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))
    print(f"[saved] {args.output}")
    if args.curve_output:
        print(f"[saved] {args.curve_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
