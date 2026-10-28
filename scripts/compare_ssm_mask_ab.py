#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare paired original and masked VSR SSM predictions."
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(
            "/home/ZhangHuayu/Workspace/VSRTask/outputs/ssm_mask_ab_checkpoint10_top5"
        ),
    )
    return parser.parse_args()


def latest_samples(run_dir):
    paths = sorted(run_dir.rglob("*samples_vsr_mask_ab_301_350.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No sample log found under {run_dir}")
    return paths[-1]


def load_samples(path):
    samples = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            score = row["score"]
            sample_id = int(score["sample_id"])
            samples[sample_id] = score
    return samples


def summarize(original, masked, start, end, require_complete=True):
    sample_ids = [i for i in range(start, end + 1) if i in original and i in masked]
    if not sample_ids:
        return None
    if require_complete and len(sample_ids) != end - start + 1:
        missing = [i for i in range(start, end + 1) if i not in original or i not in masked]
        raise ValueError(f"Missing paired samples in range {start}-{end}: {missing}")

    original_correct = sum(original[i]["accuracy"] for i in sample_ids)
    masked_correct = sum(masked[i]["accuracy"] for i in sample_ids)
    improved = [i for i in sample_ids if not original[i]["accuracy"] and masked[i]["accuracy"]]
    regressed = [i for i in sample_ids if original[i]["accuracy"] and not masked[i]["accuracy"]]
    unchanged_correct = [i for i in sample_ids if original[i]["accuracy"] and masked[i]["accuracy"]]
    unchanged_wrong = [i for i in sample_ids if not original[i]["accuracy"] and not masked[i]["accuracy"]]

    count = len(sample_ids)
    return {
        "range": f"{start}-{end}",
        "samples": count,
        "original_correct": int(original_correct),
        "original_accuracy": 100.0 * original_correct / count,
        "masked_correct": int(masked_correct),
        "masked_accuracy": 100.0 * masked_correct / count,
        "accuracy_delta_points": 100.0 * (masked_correct - original_correct) / count,
        "wrong_to_correct": improved,
        "correct_to_wrong": regressed,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def main():
    args = parse_args()
    original_path = latest_samples(args.output_base / "original")
    masked_path = latest_samples(args.output_base / "masked_top5")
    original = load_samples(original_path)
    masked = load_samples(masked_path)

    report = {
        "original_samples_file": str(original_path),
        "masked_samples_file": str(masked_path),
        "available_301_350": summarize(
            original, masked, 301, 350, require_complete=False
        ),
        "independent_311_350": summarize(original, masked, 311, 350),
    }
    report_path = args.output_base / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
