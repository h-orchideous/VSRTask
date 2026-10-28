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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.kv_feature_cache import (
    feature_cache_key,
    load_feature_cache,
    normalize_sample_id,
)
from vsrtask.patch_diff_detector import (
    TemporalPatchDiffDetector,
    fuse_qwen_layers,
)
from vsrtask.detector_data import load_diff_records, make_windows_for_sample
from train_patch_detector import (
    recover_one_edited_frame,
    resolve_clean_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize predicted and oracle patch-diff masks."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--kv-cache-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
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
            raise FileNotFoundError(f"Missing kv_diff records: {sorted(wanted - found)}")
        return records
    return load_diff_records(args.kv_diff_root, args.sample_range, None)


def build_detector(checkpoint: dict[str, Any], device: torch.device) -> TemporalPatchDiffDetector:
    state = checkpoint["detector"]
    backbone = checkpoint["model_summary"]["backbone"]
    expected_input_dim = 2 * int(backbone["kv_feature_dim"])
    if (
        "input_proj.weight" not in state
        or state["input_proj.weight"].shape[1] != expected_input_dim
    ):
        raise ValueError(
            "Checkpoint input projection is incompatible with its backbone metadata: "
            f"expected input width {expected_input_dim}."
        )
    summary = checkpoint["model_summary"]["detector"]
    detector = TemporalPatchDiffDetector(
        num_qwen_layers=int(backbone["layers"]),
        kv_dim=int(backbone["kv_feature_dim"]),
        d_model=int(summary["detector_hidden_size"]),
        num_layers=int(summary["detector_layers"]),
        num_heads=int(summary["detector_heads"]),
        max_window=int(summary["window_size"]),
        max_visual_tokens=int(summary.get("visual_tokens", 391)),
    ).to(device)
    detector.load_state_dict(state, strict=True)
    detector.eval()
    return detector


def infer_patch_grid(patches: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    aspect = width / max(height, 1)
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, int(patches**0.5) + 1):
        if patches % rows == 0:
            cols = patches // rows
            candidates.append((abs(cols / rows - aspect), rows, cols))
            candidates.append((abs(rows / cols - aspect), cols, rows))
    if not candidates:
        raise ValueError(f"Cannot infer a 2D grid for {patches} patches")
    _, rows, cols = min(candidates)
    return rows, cols


def colorize(score: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(score / max(scale, 1e-8), 0.0, 1.0)
    red = np.clip(value * 2.0, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(value * 4.0 - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 - value * 4.0, 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def resize_score(score: torch.Tensor, image_size: tuple[int, int], scale: float) -> Image.Image:
    rows, cols = infer_patch_grid(score.numel(), image_size)
    grid = score.float().cpu().reshape(rows, cols).numpy()
    heatmap = Image.fromarray(colorize(grid, scale), mode="RGB")
    return heatmap.resize(image_size, Image.Resampling.BILINEAR)


def overlay(image: Image.Image, heatmap: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)


def draw_top_patches(image: Image.Image, score: torch.Tensor, fraction: float) -> Image.Image:
    result = image.convert("RGB").copy()
    rows, cols = infer_patch_grid(score.numel(), result.size)
    keep = max(1, round(score.numel() * fraction))
    indices = torch.topk(score.float(), keep).indices.tolist()
    draw = ImageDraw.Draw(result)
    cell_w = result.width / cols
    cell_h = result.height / rows
    for index in indices:
        row, col = divmod(index, cols)
        box = (
            round(col * cell_w),
            round(row * cell_h),
            round((col + 1) * cell_w),
            round((row + 1) * cell_h),
        )
        draw.rectangle(box, outline=(255, 0, 0), width=max(2, result.width // 400))
    return result


def labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    header = 32
    canvas = Image.new("RGB", (size[0], size[1] + header), "white")
    canvas.paste(panel, (0, header))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black", font=ImageFont.load_default())
    return canvas


def save_montage(panels: list[tuple[str, Image.Image]], path: Path) -> None:
    panel_size = (360, 270)
    rendered = [labeled_panel(image, label, panel_size) for label, image in panels]
    canvas = Image.new("RGB", (panel_size[0] * len(rendered), panel_size[1] + 32), "white")
    for index, panel in enumerate(rendered):
        canvas.paste(panel, (index * panel_size[0], 0))
    canvas.save(path)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
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
    records = select_records(args)
    items: list[dict[str, Any]] = []

    with torch.inference_mode():
        for record_path in records:
            record = torch.load(record_path, map_location="cpu", weights_only=False)
            sample_id = normalize_sample_id(record["sample_id"])
            source_video = resolve_clean_video(Path(record["source_video"]), None)
            cache_path = args.kv_cache_root / f"{sample_id}.pt"
            feature_cache, cache_payload = load_feature_cache(cache_path)
            windows = make_windows_for_sample(
                record,
                args.frames_root,
                detector.max_window,
                negative_windows=0,
                seed=42,
                frame_count=int(cache_payload["frame_count"]),
            )
            for window in windows:
                k = torch.stack([
                    feature_cache[feature_cache_key(path)][0] for path in window.frame_paths
                ]).unsqueeze(0).to(device=device, dtype=dtype)
                v = torch.stack([
                    feature_cache[feature_cache_key(path)][1] for path in window.frame_paths
                ]).unsqueeze(0).to(device=device, dtype=dtype)
                predicted_layers = detector(k, v)[0].float().cpu()
                predicted_layers = predicted_layers.clamp_min(0)
                target_layers = window.target.float().cpu()
                predicted = fuse_qwen_layers(
                    predicted_layers,
                    mode=args.layer_fusion,
                    last_n_layers=args.last_n_layers,
                )
                target = fuse_qwen_layers(
                    target_layers,
                    mode=args.layer_fusion,
                    last_n_layers=args.last_n_layers,
                )
                items.append({
                    "sample_id": sample_id,
                    "frame_idx": window.current_frame_idx,
                    "edited_path": window.frame_paths[-1],
                    "clean_path": args.frames_root / sample_id / f"{window.current_frame_idx + 1:06d}.png",
                    "source_video": source_video,
                    "predicted": predicted,
                    "target": target,
                })
                print(
                    f"[predict] sample={sample_id} frame={window.current_frame_idx} "
                    f"pred_mean={predicted.mean():.6f} gt_mean={target.mean():.6f}",
                    flush=True,
                )
                if len(items) >= args.max_events:
                    break
            if len(items) >= args.max_events:
                break

    if not items:
        raise ValueError("No edited events selected")
    pred_values = torch.cat([item["predicted"].flatten() for item in items])
    target_values = torch.cat([item["target"].flatten() for item in items])
    pred_scale = float(torch.quantile(pred_values, 0.99).clamp_min(1e-8))
    target_scale = float(torch.quantile(target_values, 0.99).clamp_min(1e-8))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for item in items:
        stem = f'{item["sample_id"]}_{item["frame_idx"]:06d}'
        event_dir = args.output_dir / stem
        event_dir.mkdir(parents=True, exist_ok=True)
        clean_path = item["clean_path"]
        if not clean_path.exists():
            clean_path = event_dir / "decoded_clean.png"
            recover_one_edited_frame(
                item["source_video"], item["frame_idx"], clean_path
            )
        clean = Image.open(clean_path).convert("RGB")
        edited = Image.open(item["edited_path"]).convert("RGB")
        if clean.size != edited.size:
            clean = clean.resize(edited.size, Image.Resampling.LANCZOS)
        pred_heatmap = resize_score(item["predicted"], edited.size, pred_scale)
        target_heatmap = resize_score(item["target"], edited.size, target_scale)
        pred_overlay = overlay(edited, pred_heatmap, args.overlay_alpha)
        target_overlay = overlay(edited, target_heatmap, args.overlay_alpha)
        pred_top = draw_top_patches(edited, item["predicted"], args.top_fraction)

        clean.save(event_dir / "clean.png")
        edited.save(event_dir / "edited.png")
        pred_heatmap.save(event_dir / "pred_heatmap.png")
        pred_overlay.save(event_dir / "pred_overlay.png")
        pred_top.save(event_dir / "pred_top_patches.png")
        target_overlay.save(event_dir / "oracle_overlay.png")
        torch.save(
            {"predicted": item["predicted"], "target": item["target"]},
            event_dir / "patch_scores.pt",
        )
        save_montage(
            [
                ("clean", clean),
                ("edited", edited),
                ("predicted overlay", pred_overlay),
                ("predicted top patches", pred_top),
                ("oracle delta overlay", target_overlay),
            ],
            event_dir / "comparison.png",
        )
        metadata.append({
            "sample_id": item["sample_id"],
            "frame_idx_zero_based": item["frame_idx"],
            "edited_path": str(item["edited_path"]),
            "clean_path": str(clean_path),
            "predicted_min": float(item["predicted"].min()),
            "predicted_mean": float(item["predicted"].mean()),
            "predicted_max": float(item["predicted"].max()),
            "target_mean": float(item["target"].mean()),
            "output_dir": str(event_dir),
        })

    summary = {
        "checkpoint": str(args.checkpoint),
        "events": len(items),
        "layer_fusion": args.layer_fusion,
        "last_n_layers": args.last_n_layers,
        "top_fraction": args.top_fraction,
        "global_predicted_q99_scale": pred_scale,
        "global_target_q99_scale": target_scale,
        "items": metadata,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {len(items)} events under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
