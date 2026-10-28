#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pandas as pd
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay saved detector Top-k visual-token masks on known edited "
            "frames and build a small VSR evaluation dataset."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--mask-alpha", type=float, default=0.3)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_sample_id(value: str | int) -> str:
    return f"{int(value):08d}"


def infer_token_grid(tokens: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    aspect = width / max(height, 1)
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, int(tokens**0.5) + 1):
        if tokens % rows == 0:
            cols = tokens // rows
            candidates.append((abs(cols / rows - aspect), rows, cols))
            candidates.append((abs(rows / cols - aspect), cols, rows))
    if not candidates:
        raise ValueError(f"Cannot infer a 2D grid for {tokens} visual tokens")
    _, rows, cols = min(candidates)
    return rows, cols


def overlay_top_tokens(
    image: Image.Image,
    scores: torch.Tensor,
    fraction: float,
    alpha: float,
    outline_width: int,
) -> tuple[Image.Image, list[int], tuple[int, int]]:
    scores = scores.float().flatten().cpu()
    rows, cols = infer_token_grid(scores.numel(), image.size)
    keep = max(1, round(scores.numel() * fraction))
    indices = torch.topk(scores, keep).indices.tolist()

    width, height = image.size
    cell_width = width / cols
    cell_height = height / rows
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill_alpha = round(alpha * 255)
    for index in indices:
        row, col = divmod(index, cols)
        box = (
            round(col * cell_width),
            round(row * cell_height),
            round((col + 1) * cell_width),
            round((row + 1) * cell_height),
        )
        draw.rectangle(
            box,
            fill=(255, 0, 0, fill_alpha),
            outline=(255, 0, 0, 255),
            width=outline_width,
        )
    result = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    return result, indices, (rows, cols)


def load_mask_scores(
    mask_root: Path,
    sample_id: str,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for score_path in sorted(mask_root.glob(f"{sample_id}_*/patch_scores.pt")):
        suffix = score_path.parent.name.removeprefix(f"{sample_id}_")
        if not suffix.isdigit():
            continue
        frame_idx = int(suffix)
        payload = torch.load(score_path, map_location="cpu", weights_only=False)
        predicted = payload.get("predicted")
        if not isinstance(predicted, torch.Tensor) or predicted.ndim != 1:
            raise ValueError(f"Invalid predicted scores in {score_path}")
        if frame_idx in result:
            raise ValueError(f"Duplicate mask for {sample_id} frame {frame_idx}")
        result[frame_idx] = predicted
    return result


def build_video(
    input_path: Path,
    output_path: Path,
    frame_scores: dict[int, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        return {
            "output_video": str(output_path),
            "status": "skipped_existing",
            "masked_frame_indices": sorted(frame_scores),
        }

    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if temporary.exists():
        temporary.unlink()

    masked_rows: list[dict[str, Any]] = []
    frame_count = 0
    with av.open(str(input_path), mode="r") as source:
        input_stream = source.streams.video[0]
        rate = input_stream.average_rate or input_stream.base_rate or Fraction(1, 1)
        with av.open(str(temporary), mode="w") as target:
            output_stream = target.add_stream("libx264", rate=rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {
                "crf": str(args.crf),
                "preset": args.preset,
            }

            for frame_idx, frame in enumerate(source.decode(input_stream)):
                image = frame.to_image().convert("RGB")
                if frame_idx in frame_scores:
                    image, indices, grid = overlay_top_tokens(
                        image,
                        frame_scores[frame_idx],
                        args.top_fraction,
                        args.mask_alpha,
                        args.outline_width,
                    )
                    masked_rows.append(
                        {
                            "frame_idx_zero_based": frame_idx,
                            "visual_tokens": int(frame_scores[frame_idx].numel()),
                            "selected_tokens": len(indices),
                            "token_grid": list(grid),
                            "selected_indices": indices,
                        }
                    )
                encoded = output_stream.encode(av.VideoFrame.from_image(image))
                for packet in encoded:
                    target.mux(packet)
                frame_count += 1

            for packet in output_stream.encode():
                target.mux(packet)

    missing = sorted(set(frame_scores) - {row["frame_idx_zero_based"] for row in masked_rows})
    if missing:
        temporary.unlink(missing_ok=True)
        raise IndexError(f"Mask frame indices exceed video length for {input_path}: {missing}")
    temporary.replace(output_path)
    return {
        "input_video": str(input_path),
        "output_video": str(output_path),
        "status": "written",
        "frame_count": frame_count,
        "fps": str(rate),
        "masked_frames": masked_rows,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be in [0, 1]")
    if args.outline_width < 0:
        raise ValueError("--outline-width must be non-negative")
    start, end = args.sample_range
    if start > end:
        raise ValueError("--sample-range START must be <= END")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = {
        normalize_sample_id(sample["sample_id"]): sample
        for sample in manifest["samples"]
        if start <= int(sample["sample_id"]) <= end
    }
    expected = {normalize_sample_id(index) for index in range(start, end + 1)}
    if set(samples) != expected:
        raise KeyError(f"Manifest is missing samples: {sorted(expected - set(samples))}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for position, sample_id in enumerate(sorted(samples), start=1):
        sample = samples[sample_id]
        frame_scores = load_mask_scores(args.mask_root, sample_id)
        expected_frames = {int(spec["frame_idx"]) for spec in sample["edit_specs"]}
        if set(frame_scores) != expected_frames:
            raise ValueError(
                f"Mask/edit-frame mismatch for {sample_id}: "
                f"masks={sorted(frame_scores)} edits={sorted(expected_frames)}"
            )
        relative_video = Path(sample["video_path"])
        input_path = args.dataset_root / relative_video
        output_path = args.output_root / relative_video
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        print(
            f"[video] {position}/{len(samples)} sample={sample_id} "
            f"masked_frames={sorted(frame_scores)}",
            flush=True,
        )
        report = build_video(input_path, output_path, frame_scores, args)
        report["sample_id"] = sample_id
        reports.append(report)
        print(f"[saved] {output_path}", flush=True)

    source_table = pd.read_parquet(args.source_parquet)
    wanted_paths = {samples[sample_id]["video_path"] for sample_id in samples}
    subset = source_table[source_table["video_path"].isin(wanted_paths)].copy()
    subset = subset.sort_values("video_path").reset_index(drop=True)
    if len(subset) != len(samples):
        found = set(subset["video_path"].tolist())
        raise ValueError(f"Parquet is missing videos: {sorted(wanted_paths - found)}")
    output_parquet = args.output_root / f"test_10mins_{start}_{end}.parquet"
    subset.to_parquet(output_parquet, index=False)

    report_path = args.output_root / "mask_build_report.json"
    report_path.write_text(
        json.dumps(
            {
                "mode": "known_edited_frames_predicted_spatial_mask",
                "warning": (
                    "This diagnostic uses ground-truth edited-frame indices. "
                    "It evaluates spatial-mask utility, not end-to-end frame detection."
                ),
                "top_fraction": args.top_fraction,
                "mask_alpha": args.mask_alpha,
                "sample_range": [start, end],
                "source_parquet": str(args.source_parquet),
                "output_parquet": str(output_parquet),
                "videos": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[saved] parquet={output_parquet}", flush=True)
    print(f"[saved] report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
