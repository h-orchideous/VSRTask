#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--sample", default=None)
    args = parser.parse_args()
    paths = sorted(args.kv_diff_root.glob("*.pt"))
    if args.sample:
        paths = [args.kv_diff_root / f"{int(args.sample):08d}.pt"]
    if not paths:
        raise FileNotFoundError("No diff files found")
    record = torch.load(paths[0], map_location="cpu", weights_only=False)
    print("sample_id:", record["sample_id"])
    print("entries:", len(record["entries"]))
    for entry in record["entries"]:
        k = entry["delta_k"]
        v = entry["delta_v"]
        print(
            "frame_idx=", entry["frame_idx"],
            "edited_frame=", entry["edited_frame"],
            "layers=", len(k),
            "k_shape=", tuple(next(iter(k.values())).shape),
            "v_shape=", tuple(next(iter(v.values())).shape),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
