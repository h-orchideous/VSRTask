#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a PatchDiff event threshold from cached windows."
    )
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metrics_at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [row for row in rows if row["window_type"] == "edited"]
    negatives = [row for row in rows if row["window_type"] == "clean"]
    true_positives = sum(float(row["event_score"]) >= threshold for row in positives)
    false_positives = sum(float(row["event_score"]) >= threshold for row in negatives)
    selected = true_positives + false_positives
    return {
        "threshold": threshold,
        "recall": true_positives / len(positives),
        "sampled_clean_fpr": false_positives / len(negatives),
        "sampled_precision": true_positives / selected if selected else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.target_recall <= 1:
        raise ValueError("--target-recall must be in (0, 1]")
    source = json.loads(args.evaluation_results.read_text(encoding="utf-8"))
    rows = source.get("rows") or []
    if not rows or any("event_score" not in row for row in rows):
        raise ValueError(
            "Evaluation rows with event_score are required. Re-run "
            "evaluate_patch_diff_detector.py with the current code."
        )
    positives = [row for row in rows if row["window_type"] == "edited"]
    negatives = [row for row in rows if row["window_type"] == "clean"]
    if not positives or not negatives:
        raise ValueError("Both edited and clean windows are required")

    thresholds = sorted(
        {float(row["event_score"]) for row in rows}, reverse=True
    )
    eligible = [
        metrics_at_threshold(rows, threshold)
        for threshold in thresholds
        if metrics_at_threshold(rows, threshold)["recall"] >= args.target_recall
    ]
    if not eligible:
        raise RuntimeError("No threshold reaches the requested recall")
    selected = max(eligible, key=lambda row: row["threshold"])
    result = {
        "schema_version": 2,
        **selected,
        "selection_rule": "highest_cached_window_threshold_meeting_target_recall",
        "target_recall": args.target_recall,
        "positive_windows": len(positives),
        "sampled_clean_windows": len(negatives),
        "calibration_source": str(args.evaluation_results),
        "checkpoint": source.get("checkpoint"),
        "score_definition": {
            "layer_fusion": source.get("layer_fusion"),
            "top_fraction": source.get("top_fraction"),
            "aggregation": "mean_of_top_fraction_patch_scores",
        },
        "calibration_scope": (
            "Cached edited windows and sampled clean windows from the training "
            "distribution. This is an initial operational threshold, not an "
            "independent held-out full-frame calibration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"[saved] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
