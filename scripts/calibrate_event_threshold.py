#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate an anomaly threshold from labeled frame evaluation."
    )
    parser.add_argument("--detection-results", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.target_recall <= 1:
        raise ValueError("--target-recall must be in (0, 1]")
    source = json.loads(args.detection_results.read_text(encoding="utf-8"))
    positives: list[float] = []
    hard_negatives: list[float] = []
    for sample in source["samples"]:
        targets = set(int(index) for index in sample["target_frames"])
        positives.extend(float(score) for score in sample["target_frame_scores"])
        for frame_idx, score in zip(
            sample["predicted_top4_frames"], sample["predicted_top4_scores"]
        ):
            if int(frame_idx) not in targets:
                hard_negatives.append(float(score))

    candidates = sorted(set(positives + hard_negatives))
    rows = []
    for threshold in candidates:
        tp = sum(score >= threshold for score in positives)
        fp = sum(score >= threshold for score in hard_negatives)
        recall = tp / len(positives)
        precision = tp / (tp + fp) if tp + fp else 0.0
        rows.append(
            {
                "threshold": threshold,
                "recall": recall,
                "hard_negative_precision": precision,
                "true_positives": tp,
                "hard_false_positives": fp,
            }
        )
    eligible = [row for row in rows if row["recall"] >= args.target_recall]
    if not eligible:
        raise RuntimeError("No threshold satisfies requested recall")
    selected = max(
        eligible,
        key=lambda row: (row["hard_negative_precision"], row["threshold"]),
    )
    report = {
        "threshold": selected["threshold"],
        "target_recall": args.target_recall,
        "measured_recall": selected["recall"],
        "hard_negative_precision": selected["hard_negative_precision"],
        "true_positives": selected["true_positives"],
        "positive_count": len(positives),
        "hard_false_positives": selected["hard_false_positives"],
        "hard_negative_count": len(hard_negatives),
        "calibration_source": str(args.detection_results),
        "calibration_scope": (
            "All labeled edited frames and only the non-edited frames that "
            "entered per-video Top-4. Full clean-frame score distributions "
            "were not retained by the original evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
