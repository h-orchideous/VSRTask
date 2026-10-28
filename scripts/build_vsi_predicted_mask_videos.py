#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import site
import sys
from collections import deque
from datetime import timedelta
from fractions import Fraction
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
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_masked_vsr_videos import overlay_top_tokens
from visualize_patch_diff_masks import build_detector
from vsrtask.patch_diff_detector import fuse_qwen_layers
from vsrtask.qwen_kv import capture_edited_kv_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build VSI-SUPER-Recall videos with detector-predicted temporal "
            "events and Top-k spatial token masks. No edit-frame labels are used."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--events-per-video", type=int, default=4)
    parser.add_argument(
        "--min-event-gap",
        type=int,
        default=0,
        help=(
            "Optional temporal suppression radius in frames. The default 0 "
            "allows adjacent frames because VSI-SUPER-Recall does not provide "
            "or guarantee a minimum distance between inserted events."
        ),
    )
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--layer-fusion", choices=("mean", "max"), default="mean")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.3)
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--save-top-frames",
        action="store_true",
        help=(
            "Save the original and overlaid Top-event frames plus a montage "
            "for manual inspection."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[torch.device, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    return device, rank, world_size


def frame_anomaly_score(spatial_score: torch.Tensor, fraction: float) -> float:
    keep = max(1, round(spatial_score.numel() * fraction))
    return float(torch.topk(spatial_score.float(), keep).values.mean())


def select_temporal_events(
    frame_rows: list[dict[str, Any]], count: int, min_gap: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(frame_rows, key=lambda item: item["event_score"], reverse=True):
        if all(abs(row["frame_idx"] - item["frame_idx"]) >= min_gap for item in selected):
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(
            f"Could select only {len(selected)}/{count} events with min gap {min_gap}"
        )
    return sorted(selected, key=lambda item: item["frame_idx"])


def score_video_frames(
    video_path: Path,
    qwen: torch.nn.Module,
    processor: Any,
    detector: torch.nn.Module,
    device: torch.device,
    qwen_dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    window_k: deque[torch.Tensor] = deque(maxlen=detector.max_window)
    window_v: deque[torch.Tensor] = deque(maxlen=detector.max_window)
    frame_rows: list[dict[str, Any]] = []
    detector_dtype = next(detector.parameters()).dtype

    with av.open(str(video_path), mode="r") as source, torch.inference_mode():
        stream = source.streams.video[0]
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
            spatial_score = fuse_qwen_layers(
                predicted_layers,
                mode=args.layer_fusion,
                last_n_layers=args.last_n_layers,
            )
            frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "event_score": frame_anomaly_score(spatial_score, args.top_fraction),
                    "patch_scores": spatial_score,
                }
            )
            if (frame_idx + 1) % 10 == 0:
                print(
                    f"[detect] video={video_path.name} frames={frame_idx + 1} "
                    f"latest_score={frame_rows[-1]['event_score']:.6f}",
                    flush=True,
                )

    return frame_rows


def predict_video_events(
    video_path: Path,
    qwen: torch.nn.Module,
    processor: Any,
    detector: torch.nn.Module,
    device: torch.device,
    qwen_dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    frame_rows = score_video_frames(
        video_path, qwen, processor, detector, device, qwen_dtype, args
    )
    selected = select_temporal_events(
        frame_rows, args.events_per_video, args.min_event_gap
    )
    return selected, len(frame_rows)


def encode_masked_video(
    input_path: Path,
    output_path: Path,
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
    visual_dir: Path | None = None,
) -> list[dict[str, Any]]:
    selected_by_frame = {item["frame_idx"]: item for item in selected}
    score_rank = {
        int(item["frame_idx"]): rank
        for rank, item in enumerate(
            sorted(selected, key=lambda item: item["event_score"], reverse=True),
            start=1,
        )
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if visual_dir is not None:
        visual_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    masks: list[dict[str, Any]] = []

    with av.open(str(input_path), mode="r") as source:
        input_stream = source.streams.video[0]
        rate = input_stream.average_rate or input_stream.base_rate or Fraction(1, 1)
        with av.open(str(temporary), mode="w") as target:
            output_stream = target.add_stream("libx264", rate=rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": str(args.crf), "preset": args.preset}

            for frame_idx, frame in enumerate(source.decode(input_stream)):
                image = frame.to_image().convert("RGB")
                if frame_idx in selected_by_frame:
                    row = selected_by_frame[frame_idx]
                    original = image.copy()
                    image, indices, grid = overlay_top_tokens(
                        image,
                        row["patch_scores"],
                        args.top_fraction,
                        args.mask_alpha,
                        args.outline_width,
                    )
                    masks.append(
                        {
                            "score_rank": score_rank[frame_idx],
                            "frame_idx": frame_idx,
                            "event_score": row["event_score"],
                            "selected_indices": indices,
                            "token_grid": list(grid),
                        }
                    )
                    if visual_dir is not None:
                        stem = (
                            f"rank{score_rank[frame_idx]:02d}_frame{frame_idx:06d}_"
                            f"score{row['event_score']:.6f}"
                        )
                        original.save(visual_dir / f"{stem}_original.jpg", quality=95)
                        image.save(visual_dir / f"{stem}_overlay.jpg", quality=95)
                for packet in output_stream.encode(av.VideoFrame.from_image(image)):
                    target.mux(packet)
            for packet in output_stream.encode():
                target.mux(packet)

    temporary.replace(output_path)
    if visual_dir is not None:
        save_top_frame_montage(visual_dir, masks)
    return masks


def save_top_frame_montage(
    visual_dir: Path,
    masks: list[dict[str, Any]],
) -> None:
    pairs: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    for item in sorted(masks, key=lambda value: value["score_rank"]):
        prefix = (
            f"rank{item['score_rank']:02d}_frame{item['frame_idx']:06d}_"
            f"score{item['event_score']:.6f}"
        )
        with Image.open(visual_dir / f"{prefix}_original.jpg") as source:
            original = source.convert("RGB")
        with Image.open(visual_dir / f"{prefix}_overlay.jpg") as source:
            overlay = source.convert("RGB")
        pairs.append((item, original, overlay))
    if not pairs:
        return

    thumb_width = 640
    label_height = 34
    resized: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    row_heights: list[int] = []
    for item, original, overlay in pairs:
        height = max(1, round(original.height * thumb_width / original.width))
        original = original.resize((thumb_width, height), Image.Resampling.LANCZOS)
        overlay = overlay.resize((thumb_width, height), Image.Resampling.LANCZOS)
        resized.append((item, original, overlay))
        row_heights.append(height + label_height)

    montage = Image.new(
        "RGB",
        (thumb_width * 2, sum(row_heights)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(montage)
    y = 0
    for item, original, overlay in resized:
        label = (
            f"rank={item['score_rank']} frame={item['frame_idx']} "
            f"score={item['event_score']:.6f}"
        )
        draw.text((8, y + 8), f"original | {label}", fill=(0, 0, 0))
        draw.text(
            (thumb_width + 8, y + 8),
            f"selected-token overlay | {label}",
            fill=(0, 0, 0),
        )
        montage.paste(original, (0, y + label_height))
        montage.paste(overlay, (thumb_width, y + label_height))
        y += original.height + label_height
    montage.save(visual_dir / "top_frames_montage.jpg", quality=95)


def main() -> int:
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    if args.events_per_video <= 0 or args.min_event_gap < 0:
        raise ValueError("events-per-video must be positive and min-event-gap nonnegative")

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
    report_root = args.output_root / "detector_reports"
    report_root.mkdir(parents=True, exist_ok=True)

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
        f"[vsi-mask] rank={rank}/{world_size} videos={len(local_rows)} "
        f"events={args.events_per_video} top_fraction={args.top_fraction}",
        flush=True,
    )
    for position, row in enumerate(local_rows, start=1):
        relative_path = Path(row["video_path"])
        input_path = args.dataset_root / relative_path
        output_path = args.output_root / relative_path
        report_path = report_root / f"{relative_path.parent.name}_{relative_path.stem}.json"
        visual_dir = args.output_root / "top_frames" / relative_path.parent / relative_path.stem
        visuals_ready = (
            not args.save_top_frames
            or (visual_dir / "top_frames_montage.jpg").exists()
        )
        if (
            output_path.exists()
            and report_path.exists()
            and visuals_ready
            and not args.overwrite
        ):
            print(f"[skip] rank={rank} {relative_path}", flush=True)
            continue
        print(
            f"[video] rank={rank} {position}/{len(local_rows)} {relative_path}",
            flush=True,
        )
        selected, frame_count = predict_video_events(
            input_path, qwen, processor, detector, device, qwen_dtype, args
        )
        masks = encode_masked_video(
            input_path,
            output_path,
            selected,
            args,
            visual_dir if args.save_top_frames else None,
        )
        report = {
            "video_path": str(relative_path),
            "question": row.get("question"),
            "input_video": str(input_path),
            "output_video": str(output_path),
            "top_frames_visual_dir": (
                str(visual_dir) if args.save_top_frames else None
            ),
            "frame_count": frame_count,
            "selection": (
                "predicted_global_top_events"
                if args.min_event_gap == 0
                else "predicted_top_events_with_temporal_nms"
            ),
            "events_per_video": args.events_per_video,
            "min_event_gap": args.min_event_gap,
            "top_fraction": args.top_fraction,
            "masked_frames": masks,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.save_top_frames:
            (visual_dir / "top_frames.json").write_text(
                json.dumps(
                    {
                        "video_path": str(relative_path),
                        "question": row.get("question"),
                        "top_fraction": args.top_fraction,
                        "predicted_top_frames": sorted(
                            masks, key=lambda item: item["score_rank"]
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            f"[saved] {output_path} frames={[item['frame_idx'] for item in masks]}",
            flush=True,
        )

    if world_size > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        output_parquet = args.output_root / args.source_parquet.name
        table.to_parquet(output_parquet, index=False)
        reports = []
        for path in sorted(report_root.glob("*.json")):
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        summary = {
            "mode": "fully_predicted_temporal_and_spatial_mask",
            "checkpoint": str(args.checkpoint),
            "source_parquet": str(args.source_parquet),
            "output_parquet": str(output_parquet),
            "videos": len(reports),
            "events_per_video": args.events_per_video,
            "min_event_gap": args.min_event_gap,
            "top_fraction": args.top_fraction,
            "reports": reports,
        }
        summary_path = args.output_root / f"mask_build_report_{args.source_parquet.stem}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[complete] videos={len(reports)} root={args.output_root}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
