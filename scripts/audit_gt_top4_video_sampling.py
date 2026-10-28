#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that each four-frame oracle MP4 preserves all source frames."
    )
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float32)


def main() -> int:
    args = parse_args()
    # Import the exact loader used by lmms-eval instead of reimplementing sampling.
    from lmms_eval.models.model_utils.load_video import read_video_pyav_pil

    build = json.loads(args.build_report.read_text(encoding="utf-8"))
    results = []
    for position, record in enumerate(build["records"], start=1):
        source_paths = [Path(path) for path in record["edited_frames_temporal_order"]]
        video_path = args.video_root / record["video_path"]
        decoded = read_video_pyav_pil(
            str(video_path),
            num_frm=4,
            fps=1.0,
            max_image_size=None,
            force_include_last_frame=True,
        )
        sources = [load_rgb(path) for path in source_paths]
        distance_matrix = []
        for decoded_image in decoded:
            decoded_array = np.asarray(decoded_image.convert("RGB"), dtype=np.float32)
            distance_matrix.append(
                [float(np.mean(np.abs(decoded_array - source))) for source in sources]
            )
        nearest_source_indices = [
            int(np.argmin(distances)) for distances in distance_matrix
        ]
        expected = list(range(4))
        result = {
            "sample_id": int(record["sample_id"]),
            "video_path": str(video_path),
            "decoded_frame_count": len(decoded),
            "expected_frame_count": 4,
            "nearest_source_indices": nearest_source_indices,
            "expected_source_indices": expected,
            "order_preserved": len(decoded) == 4 and nearest_source_indices == expected,
            "matched_frame_mae": [
                distance_matrix[index][source_index]
                for index, source_index in enumerate(nearest_source_indices)
            ],
            "distance_matrix": distance_matrix,
        }
        results.append(result)
        print(
            f"[audit] {position}/{len(build['records'])} "
            f"id={result['sample_id']:08d} decoded={len(decoded)} "
            f"mapping={nearest_source_indices} ok={result['order_preserved']}",
            flush=True,
        )

    passed = sum(bool(result["order_preserved"]) for result in results)
    summary = {
        "loader": "lmms_eval.models.model_utils.load_video.read_video_pyav_pil",
        "loader_arguments": {
            "num_frm": 4,
            "fps": 1.0,
            "force_include_last_frame": True,
        },
        "samples": len(results),
        "all_four_frames_and_order_preserved": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[complete] passed={passed}/{len(results)} output={args.output_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
