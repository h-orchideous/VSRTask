#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path
from typing import Any

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.anomaly_detector import (
    TemporalPatchAnomalyDetector,
    diff_patch_magnitude,
    fuse_layer_diff,
)
from vsrtask.detector_data import load_diff_records, make_windows_for_sample
from train_patch_detector import (
    feature_cache_key,
    load_feature_cache,
    normalize_sample_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained temporal patch detector."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--kv-cache-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int, default=None)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--negative-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_records(args: argparse.Namespace) -> list[Path]:
    if args.sample_ids and args.sample_range:
        raise ValueError("Use either --sample-ids or --sample-range")
    if args.sample_ids:
        wanted = {normalize_sample_id(value) for value in args.sample_ids}
        paths = [
            path for path in sorted(args.kv_diff_root.glob("*.pt"))
            if normalize_sample_id(path.stem) in wanted
        ]
        found = {normalize_sample_id(path.stem) for path in paths}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"Missing kv_diff records: {missing}")
        return paths
    return load_diff_records(args.kv_diff_root, args.sample_range, None)


def build_detector(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> TemporalPatchAnomalyDetector:
    backbone = checkpoint["model_summary"]["backbone"]
    summary = checkpoint["model_summary"]["detector"]
    detector = TemporalPatchAnomalyDetector(
        num_qwen_layers=int(backbone["layers"]),
        kv_dim=int(backbone["kv_feature_dim"]),
        d_model=int(summary["detector_hidden_size"]),
        num_layers=int(summary["detector_layers"]),
        num_heads=int(summary["detector_heads"]),
        max_window=int(summary["window_size"]),
    ).to(device)
    detector.load_state_dict(checkpoint["detector"], strict=True)
    detector.eval()
    return detector


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    detector = build_detector(checkpoint, device)

    records = select_records(args)
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for record_path in records:
            record = torch.load(
                record_path,
                map_location="cpu",
                weights_only=False,
            )
            sample_id = normalize_sample_id(record["sample_id"])
            cache_path = args.kv_cache_root / f"{sample_id}.pt"
            feature_cache, cache_payload = load_feature_cache(cache_path)
            frame_count = int(cache_payload["frame_count"])
            windows = make_windows_for_sample(
                record,
                args.frames_root,
                int(detector.max_window),
                args.negative_windows,
                args.seed,
                frame_count=frame_count,
            )

            for window in windows:
                k = torch.stack(
                    [
                        feature_cache[feature_cache_key(path)][0]
                        for path in window.frame_paths
                    ]
                ).unsqueeze(0).to(device=device, dtype=next(detector.parameters()).dtype)
                v = torch.stack(
                    [
                        feature_cache[feature_cache_key(path)][1]
                        for path in window.frame_paths
                    ]
                ).unsqueeze(0).to(device=device, dtype=next(detector.parameters()).dtype)
                target_k = window.target_k.unsqueeze(0).to(device)
                target_v = window.target_v.unsqueeze(0).to(device)
                prediction_k, prediction_v = detector(k, v)
                loss_k = nn.functional.mse_loss(prediction_k, target_k.float())
                loss_v = nn.functional.mse_loss(prediction_v, target_v.float())
                loss = 0.5 * (loss_k + loss_v)
                prediction = diff_patch_magnitude(
                    prediction_k[0], prediction_v[0]
                )
                target = diff_patch_magnitude(target_k[0], target_v[0])
                fused_prediction = fuse_layer_diff(prediction, mode="mean")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "current_frame_idx": window.current_frame_idx,
                        "window_type": "edited" if window.positive else "clean",
                        "loss": float(loss.cpu()),
                        "target_diff_mean": float(target.mean().cpu()),
                        "predicted_diff_mean": float(prediction.mean().cpu()),
                        "target_k_mean": float(target_k.mean().cpu()),
                        "target_v_mean": float(target_v.mean().cpu()),
                        "predicted_k_mean": float(prediction_k.mean().cpu()),
                        "predicted_v_mean": float(prediction_v.mean().cpu()),
                        "loss_k": float(loss_k.cpu()),
                        "loss_v": float(loss_v.cpu()),
                        "fused_prediction_mean": float(
                            fused_prediction.mean().cpu()
                        ),
                        "fused_prediction_max": float(
                            fused_prediction.max().cpu()
                        ),
                    }
                )
            print(
                f"[evaluate] sample={sample_id} windows={len(windows)}",
                flush=True,
            )

    def mean(items: list[dict[str, Any]], key: str) -> float | None:
        return (
            sum(float(item[key]) for item in items) / len(items)
            if items else None
        )

    positive = [row for row in rows if row["window_type"] == "edited"]
    negative = [row for row in rows if row["window_type"] == "clean"]
    metrics = {
        "overall_mse": mean(rows, "loss"),
        "positive_mse": mean(positive, "loss"),
        "negative_mse": mean(negative, "loss"),
        "positive_predicted_diff_mean": mean(
            positive, "predicted_diff_mean"
        ),
        "negative_predicted_diff_mean": mean(
            negative, "predicted_diff_mean"
        ),
        "positive_predicted_k_mean": mean(positive, "predicted_k_mean"),
        "negative_predicted_k_mean": mean(negative, "predicted_k_mean"),
        "positive_predicted_v_mean": mean(positive, "predicted_v_mean"),
        "negative_predicted_v_mean": mean(negative, "predicted_v_mean"),
        "positive_fused_prediction_mean": mean(
            positive, "fused_prediction_mean"
        ),
        "negative_fused_prediction_mean": mean(
            negative, "fused_prediction_mean"
        ),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "samples": len(records),
        "windows": len(rows),
        "positive_windows": len(positive),
        "negative_windows": len(negative),
        "metrics": metrics,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
