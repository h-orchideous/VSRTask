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
        default=Path("/home/ZhangHuayu/Workspace/VSRTask/outputs/ssm_patchdiff_ab"),
    )
    return parser.parse_args()


def latest_samples(root: Path, task: str) -> Path:
    candidates = sorted(root.rglob(f"*samples_{task}.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No samples for {task} under {root}")
    return candidates[-1]


def load_samples(path: Path):
    samples = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)["score"]
            samples[int(Path(row["video_path"]).stem)] = row
    return samples


def main() -> int:
    args = parse_args()
    task = f"cambrians_vsr_local_{args.duration}"
    root = args.output_base / args.duration
    baseline_path = latest_samples(root / "ssm", task)
    patchdiff_path = latest_samples(root / "ssm_patchdiff", task)
    baseline = load_samples(baseline_path)
    patchdiff = load_samples(patchdiff_path)
    ids = sorted(set(baseline) & set(patchdiff))
    if len(ids) != 60:
        raise ValueError(f"Expected 60 paired samples, got {len(ids)}")

    baseline_correct = sum(baseline[index]["accuracy"] for index in ids)
    patchdiff_correct = sum(patchdiff[index]["accuracy"] for index in ids)
    improved = [
        index for index in ids
        if not baseline[index]["accuracy"] and patchdiff[index]["accuracy"]
    ]
    regressed = [
        index for index in ids
        if baseline[index]["accuracy"] and not patchdiff[index]["accuracy"]
    ]
    report = {
        "duration": args.duration,
        "samples": len(ids),
        "baseline_samples_file": str(baseline_path),
        "patchdiff_samples_file": str(patchdiff_path),
        "ssm_correct": int(baseline_correct),
        "ssm_accuracy": 100.0 * baseline_correct / len(ids),
        "ssm_patchdiff_correct": int(patchdiff_correct),
        "ssm_patchdiff_accuracy": 100.0 * patchdiff_correct / len(ids),
        "accuracy_delta_points": (
            100.0 * (patchdiff_correct - baseline_correct) / len(ids)
        ),
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
