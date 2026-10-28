#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default="10mins")
    parser.add_argument(
        "--ab-root",
        type=Path,
        default=Path(
            "/home/ZhangHuayu/Workspace/VSRTask/outputs/"
            "vsi_super_recall_ssm_ab_checkpoint10_top5"
        ),
    )
    parser.add_argument(
        "--top4-root",
        type=Path,
        default=Path(
            "/home/ZhangHuayu/Workspace/VSRTask/outputs/"
            "vsi_super_recall_patchdiff_top4_raw_eval"
        ),
    )
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=Path(
            "/home/ZhangHuayu/Workspace/VSRTask/outputs/"
            "vsi_super_recall_patchdiff_top4_raw_qwen_eval"
        ),
    )
    return parser.parse_args()


def latest_samples(root: Path, task: str) -> Path:
    paths = sorted(root.rglob(f"*samples_{task}.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {task} samples under {root}")
    return paths[-1]


def load(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)["score"]
            rows[int(Path(row["video_path"]).stem)] = row
    return rows


def score(rows, ids):
    correct = int(sum(rows[index]["accuracy"] for index in ids))
    return correct, 100.0 * correct / len(ids)


def main() -> int:
    args = parse_args()
    task = f"cambrians_vsr_local_{args.duration}"
    ab = args.ab_root / args.duration
    original_path = latest_samples(ab / "original", task)
    masked_path = latest_samples(ab / "masked_top5", task)
    top4_ssm_path = latest_samples(args.top4_root / args.duration, task)
    top4_qwen_path = latest_samples(args.qwen_root / args.duration, task)
    original = load(original_path)
    masked = load(masked_path)
    top4_ssm = load(top4_ssm_path)
    top4_qwen = load(top4_qwen_path)
    ids = sorted(set(original) & set(masked) & set(top4_ssm) & set(top4_qwen))
    if len(ids) != 60:
        raise ValueError(f"Expected 60 paired samples, got {len(ids)}")

    original_correct, original_accuracy = score(original, ids)
    masked_correct, masked_accuracy = score(masked, ids)
    top4_ssm_correct, top4_ssm_accuracy = score(top4_ssm, ids)
    top4_qwen_correct, top4_qwen_accuracy = score(top4_qwen, ids)
    report = {
        "duration": args.duration,
        "samples": len(ids),
        "original": {"correct": original_correct, "accuracy": original_accuracy},
        "masked_top5": {"correct": masked_correct, "accuracy": masked_accuracy},
        "patchdiff_top4_raw_ssm": {
            "correct": top4_ssm_correct,
            "accuracy": top4_ssm_accuracy,
        },
        "patchdiff_top4_raw_qwen": {
            "correct": top4_qwen_correct,
            "accuracy": top4_qwen_accuracy,
        },
        "qwen_delta_vs_top4_ssm_points": top4_qwen_accuracy - top4_ssm_accuracy,
        "qwen_delta_vs_full_video_ssm_points": top4_qwen_accuracy - original_accuracy,
        "qwen_wrong_to_correct_vs_top4_ssm": [
            index for index in ids
            if not top4_ssm[index]["accuracy"] and top4_qwen[index]["accuracy"]
        ],
        "qwen_correct_to_wrong_vs_top4_ssm": [
            index for index in ids
            if top4_ssm[index]["accuracy"] and not top4_qwen[index]["accuracy"]
        ],
        "files": {
            "original": str(original_path),
            "masked_top5": str(masked_path),
            "patchdiff_top4_raw_ssm": str(top4_ssm_path),
            "patchdiff_top4_raw_qwen": str(top4_qwen_path),
        },
    }
    output_path = args.qwen_root / args.duration / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[saved] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
