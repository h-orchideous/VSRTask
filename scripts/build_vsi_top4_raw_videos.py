#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import av
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build four-frame VSR videos from PatchDiff-selected original "
            "frames. No mask or pixel overlay is applied."
        )
    )
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--detector-report-root", type=Path, required=True)
    parser.add_argument("--top-frames-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def original_frame_path(root: Path, frame_idx: int) -> Path:
    matches = list(root.glob(f"rank*_frame{frame_idx:06d}_score*_original.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one original image for frame {frame_idx} under {root}, "
            f"found {matches}"
        )
    return matches[0]


def encode_video(
    images: list[Image.Image],
    output_path: Path,
    crf: int,
    preset: str,
) -> None:
    if len(images) != 4:
        raise ValueError(f"Expected four images, got {len(images)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    with av.open(str(temporary), mode="w") as target:
        stream = target.add_stream("libx264", rate=Fraction(1, 1))
        stream.width = images[0].width
        stream.height = images[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        for index, image in enumerate(images):
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            frame.time_base = Fraction(1, 1)
            for packet in stream.encode(frame):
                target.mux(packet)
        for packet in stream.encode():
            target.mux(packet)
    temporary.replace(output_path)


def main() -> int:
    args = parse_args()
    table = pd.read_parquet(args.source_parquet).sort_values("video_path")
    if len(table) != 60:
        raise ValueError(f"Expected 60 official samples, got {len(table)}")

    build_rows = []
    for position, record in enumerate(table.to_dict("records"), start=1):
        relative_path = Path(record["video_path"])
        report_name = f"{relative_path.parent.name}_{relative_path.stem}.json"
        report_path = args.detector_report_root / report_name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = sorted(
            report["masked_frames"], key=lambda item: int(item["frame_idx"])
        )
        frame_indices = [int(item["frame_idx"]) for item in selected]
        if len(frame_indices) != 4 or len(set(frame_indices)) != 4:
            raise ValueError(f"Invalid Top-4 for {relative_path}: {frame_indices}")

        output_path = args.output_root / relative_path
        if not output_path.is_file() or args.overwrite:
            image_root = args.top_frames_root / relative_path.parent / relative_path.stem
            images = []
            for frame_idx in frame_indices:
                with Image.open(original_frame_path(image_root, frame_idx)) as source:
                    images.append(source.convert("RGB"))
            encode_video(images, output_path, args.crf, args.preset)
            status = "saved"
        else:
            status = "skip"
        build_rows.append(
            {
                "video_path": str(relative_path),
                "output_video": str(output_path),
                "selected_frames_temporal_order": frame_indices,
            }
        )
        print(
            f"[{status}] {position}/60 {relative_path} frames={frame_indices}",
            flush=True,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_parquet = args.output_root / args.source_parquet.name
    table.to_parquet(output_parquet, index=False)
    report = {
        "mode": "patchdiff_top4_original_frames_without_mask",
        "videos": len(build_rows),
        "frames_per_video": 4,
        "frame_rate": 1,
        "source_parquet": str(args.source_parquet),
        "output_parquet": str(output_parquet),
        "samples": build_rows,
    }
    (args.output_root / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[complete] videos=60 root={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
