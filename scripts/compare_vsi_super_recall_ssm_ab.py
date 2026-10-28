#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default="10mins")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(
            "/home/ZhangHuayu/Workspace/VSRTask/outputs/"
            "vsi_super_recall_ssm_ab_checkpoint10_top5"
        ),
    )
    return parser.parse_args()


def latest_samples(run_dir: Path, task: str) -> Path:
    paths = sorted(run_dir.rglob(f"*samples_{task}.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {task} sample log under {run_dir}")
    return paths[-1]


def load(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)["score"]
            sample_id = int(Path(row["video_path"]).stem)
            rows[sample_id] = row
    return rows


def main():
    args = parse_args()
    task = f"cambrians_vsr_local_{args.duration}"
    root = args.output_base / args.duration
    original_path = latest_samples(root / "original", task)
    masked_path = latest_samples(root / "masked_top5", task)
    original = load(original_path)
    masked = load(masked_path)
    ids = sorted(set(original) & set(masked))
    if len(ids) != 60:
        raise ValueError(f"Expected 60 paired samples, got {len(ids)}")

    original_correct = sum(original[i]["accuracy"] for i in ids)
    masked_correct = sum(masked[i]["accuracy"] for i in ids)
    improved = [i for i in ids if not original[i]["accuracy"] and masked[i]["accuracy"]]
    regressed = [i for i in ids if original[i]["accuracy"] and not masked[i]["accuracy"]]
    report = {
        "duration": args.duration,
        "samples": len(ids),
        "original_samples_file": str(original_path),
        "masked_samples_file": str(masked_path),
        "original_correct": int(original_correct),
        "original_accuracy": 100.0 * original_correct / len(ids),
        "masked_correct": int(masked_correct),
        "masked_accuracy": 100.0 * masked_correct / len(ids),
        "accuracy_delta_points": 100.0 * (masked_correct - original_correct) / len(ids),
        "wrong_to_correct": improved,
        "correct_to_wrong": regressed,
    }
    output_path = root / "comparison.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
