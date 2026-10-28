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
        description="Build four-frame videos from ground-truth edited VSR frames."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-range", type=int, nargs=2, default=(301, 350))
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_video(
    image_paths: list[Path], output_path: Path, crf: int, preset: str
) -> None:
    if len(image_paths) != 4:
        raise ValueError(f"Expected four edited frames, got {len(image_paths)}")
    images = []
    for path in image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError(f"Edited frames have inconsistent sizes: {image_paths}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    with av.open(str(temporary), mode="w") as target:
        stream = target.add_stream("libx264", rate=Fraction(1, 1))
        stream.width, stream.height = size
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
    start, end = args.sample_range
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for sample in manifest["samples"]:
        sample_id = int(sample["sample_id"])
        if start <= sample_id <= end:
            records.append(sample)
    records.sort(key=lambda row: int(row["sample_id"]))
    expected = end - start + 1
    if len(records) != expected:
        raise ValueError(f"Expected {expected} samples, found {len(records)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    parquet_rows = []
    report_rows = []
    for position, sample in enumerate(records, start=1):
        entries = sorted(sample["edit_specs"], key=lambda row: int(row["frame_idx"]))
        if len(entries) != 4:
            raise ValueError(
                f"Sample {sample['sample_id']} has {len(entries)} edited frames"
            )
        frame_indices = [int(entry["frame_idx"]) for entry in entries]
        image_paths = [Path(entry["edited_frame"]) for entry in entries]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing edited frames for sample {sample['sample_id']}: {missing}"
            )

        relative_video = Path(sample["video_path"])
        output_video = args.output_root / relative_video
        if args.overwrite or not output_video.is_file():
            encode_video(image_paths, output_video, args.crf, args.preset)
            status = "saved"
        else:
            status = "skip"

        parquet_rows.append(
            {
                "video_path": str(relative_video),
                "question": sample["question"],
                "options": sample["options"],
                "answer": sample["answer"],
                "type": sample["type"],
            }
        )
        report_rows.append(
            {
                "sample_id": sample["sample_id"],
                "video_path": str(relative_video),
                "frame_indices_temporal_order": frame_indices,
                "edited_frames_temporal_order": [str(path) for path in image_paths],
                "locations_temporal_order": [entry["location"] for entry in entries],
                "answer": sample["answer"],
            }
        )
        print(
            f"[{status}] {position}/{expected} sample={sample['sample_id']} "
            f"frames={frame_indices}",
            flush=True,
        )

    parquet_path = args.output_root / f"test_10mins_{start}_{end}.parquet"
    pd.DataFrame(parquet_rows).to_parquet(parquet_path, index=False)
    report = {
        "mode": "ground_truth_edited_frames_in_temporal_order",
        "sample_range": [start, end],
        "samples": expected,
        "frames_per_video": 4,
        "frame_rate": 1,
        "manifest": str(args.manifest),
        "parquet": str(parquet_path),
        "records": report_rows,
    }
    report_path = args.output_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[complete] videos={expected} parquet={parquet_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
