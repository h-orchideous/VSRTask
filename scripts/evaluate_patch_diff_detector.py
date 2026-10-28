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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.detector_data import load_diff_records, make_windows_for_sample
from vsrtask.kv_feature_cache import (
    feature_cache_key,
    load_feature_cache,
    normalize_sample_id,
)
from vsrtask.patch_diff_detector import (
    TemporalPatchDiffDetector,
    fuse_qwen_layers,
    patch_diff_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a patch-diff detector.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--kv-cache-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--negative-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_records(args: argparse.Namespace) -> list[Path]:
    if args.sample_ids and args.sample_range:
        raise ValueError("Use either --sample-ids or --sample-range")
    if args.sample_ids:
        wanted = {normalize_sample_id(value) for value in args.sample_ids}
        records = [
            path for path in sorted(args.kv_diff_root.glob("*.pt"))
            if normalize_sample_id(path.stem) in wanted
        ]
        found = {normalize_sample_id(path.stem) for path in records}
        if found != wanted:
            raise FileNotFoundError(f"Missing records: {sorted(wanted - found)}")
        return records
    return load_diff_records(
        args.kv_diff_root,
        tuple(args.sample_range) if args.sample_range else None,
        None,
    )


def build_detector(
    checkpoint: dict[str, Any], device: torch.device
) -> TemporalPatchDiffDetector:
    summary = checkpoint["model_summary"]
    backbone = summary["backbone"]
    detector_summary = summary["detector"]
    detector = TemporalPatchDiffDetector(
        num_qwen_layers=int(backbone["layers"]),
        kv_dim=int(backbone["kv_feature_dim"]),
        d_model=int(detector_summary["detector_hidden_size"]),
        num_layers=int(detector_summary["detector_layers"]),
        num_heads=int(detector_summary["detector_heads"]),
        max_window=int(detector_summary["window_size"]),
        max_visual_tokens=int(detector_summary.get("visual_tokens", 391)),
    ).to(device)
    detector.load_state_dict(checkpoint["detector"], strict=True)
    detector.eval()
    return detector


def topk_iou(prediction: torch.Tensor, target: torch.Tensor, fraction: float) -> float:
    keep = max(1, round(target.numel() * fraction))
    predicted_indices = set(torch.topk(prediction.flatten(), keep).indices.tolist())
    target_indices = set(torch.topk(target.flatten(), keep).indices.tolist())
    intersection = len(predicted_indices & target_indices)
    union = len(predicted_indices | target_indices)
    return intersection / max(union, 1)


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def main() -> int:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top-fraction must be in (0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    detector = build_detector(checkpoint, device)
    dtype = next(detector.parameters()).dtype
    rows: list[dict[str, Any]] = []
    records = select_records(args)

    with torch.inference_mode():
        for record_path in records:
            record = torch.load(record_path, map_location="cpu", weights_only=False)
            sample_id = normalize_sample_id(record["sample_id"])
            cache_path = args.kv_cache_root / f"{sample_id}.pt"
            feature_cache, cache_payload = load_feature_cache(cache_path)
            windows = make_windows_for_sample(
                record,
                args.frames_root,
                detector.max_window,
                args.negative_windows,
                args.seed,
                frame_count=int(cache_payload["frame_count"]),
                require_edited_frames=False,
            )
            for window in windows:
                k = torch.stack([
                    feature_cache[feature_cache_key(path)][0]
                    for path in window.frame_paths
                ]).unsqueeze(0).to(device=device, dtype=dtype)
                v = torch.stack([
                    feature_cache[feature_cache_key(path)][1]
                    for path in window.frame_paths
                ]).unsqueeze(0).to(device=device, dtype=dtype)
                target = patch_diff_target(
                    window.target_k, window.target_v
                ).unsqueeze(0).to(device)
                prediction = detector(k, v)
                loss = (prediction.float() - target.float()).square().mean()
                predicted_map = fuse_qwen_layers(
                    prediction[0].float().clamp_min(0).cpu(),
                    args.layer_fusion,
                    args.last_n_layers,
                )
                target_map = fuse_qwen_layers(
                    target[0].float().cpu(),
                    args.layer_fusion,
                    args.last_n_layers,
                )
                row = {
                    "sample_id": sample_id,
                    "current_frame_idx": window.current_frame_idx,
                    "window_type": "edited" if window.positive else "clean",
                    "mse": float(loss.cpu()),
                    "predicted_diff_mean": float(prediction.float().mean().cpu()),
                    "predicted_nonnegative_mean": float(prediction.float().clamp_min(0).mean().cpu()),
                    "target_diff_mean": float(target.float().mean().cpu()),
                    "fused_prediction_mean": float(predicted_map.mean()),
                    "fused_prediction_max": float(predicted_map.max()),
                    "topk_iou": (
                        topk_iou(predicted_map, target_map, args.top_fraction)
                        if window.positive else None
                    ),
                }
                rows.append(row)
            print(f"[evaluate] sample={sample_id} windows={len(windows)}", flush=True)

    positive = [row for row in rows if row["window_type"] == "edited"]
    negative = [row for row in rows if row["window_type"] == "clean"]
    metrics = {
        "overall_mse": mean(rows, "mse"),
        "positive_mse": mean(positive, "mse"),
        "negative_mse": mean(negative, "mse"),
        "positive_target_diff_mean": mean(positive, "target_diff_mean"),
        "positive_predicted_diff_mean": mean(positive, "predicted_nonnegative_mean"),
        "negative_predicted_diff_mean": mean(negative, "predicted_nonnegative_mean"),
        "positive_fused_prediction_mean": mean(positive, "fused_prediction_mean"),
        "negative_fused_prediction_mean": mean(negative, "fused_prediction_mean"),
        "positive_topk_iou": mean(positive, "topk_iou"),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "samples": len(records),
        "positive_windows": len(positive),
        "negative_windows": len(negative),
        "layer_fusion": args.layer_fusion,
        "last_n_layers": args.last_n_layers,
        "top_fraction": args.top_fraction,
        "metrics": metrics,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
