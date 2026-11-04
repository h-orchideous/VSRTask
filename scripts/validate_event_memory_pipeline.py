#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vsrtask.semantic_memory import semantic_memory_complete


FORBIDDEN_MEMORY_KEYS = {"question", "options", "answer", "target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate event-memory pipeline artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--semantic-root",
        type=Path,
        help="Semantic memory root; defaults to RUN_ROOT/semantic/memories.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        help="QA results path; defaults to RUN_ROOT/qa/results.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Validation report path; defaults to RUN_ROOT/validation_report.json.",
    )
    qa_mode = parser.add_mutually_exclusive_group()
    qa_mode.add_argument("--require-qa", action="store_true")
    qa_mode.add_argument("--skip-qa", action="store_true")
    return parser.parse_args()


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(collect_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(collect_keys(child))
        return keys
    return set()


def memory_set_digest(memory_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(memory_root.rglob("*.json"))
    for path in paths:
        digest.update(str(path.relative_to(memory_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    detection_root = args.run_root / "detection" / "memories"
    semantic_root = args.semantic_root or args.run_root / "semantic" / "memories"
    qa_path = args.qa_path or args.run_root / "qa" / "results.json"
    detection_paths = sorted(detection_root.rglob("*.json"))
    if not detection_paths:
        raise FileNotFoundError(f"No detection memories under {detection_root}")

    errors = []
    event_total = 0
    semantic_event_total = 0
    for detection_path in detection_paths:
        relative = detection_path.relative_to(detection_root)
        detection = json.loads(detection_path.read_text(encoding="utf-8"))
        events = detection.get("events", [])
        event_total += len(events)
        frames = [int(event["peak_frame"]) for event in events]
        if frames != sorted(frames):
            errors.append(f"{relative}: detection events are not time ordered")
        if detection.get("detector", {}).get("merge_gap") != -1:
            errors.append(f"{relative}: merge_gap is not disabled")
        for event in events:
            evidence = event.get("evidence", {})
            for key in ("original_image", "overlay_image"):
                path_value = evidence.get(key)
                if not path_value or not Path(path_value).is_file():
                    errors.append(f"{relative}: missing {key} for event {event.get('event_id')}")

        semantic_path = semantic_root / relative
        if not semantic_path.is_file():
            errors.append(f"{relative}: missing semantic memory")
            continue
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        leaked = collect_keys(semantic) & FORBIDDEN_MEMORY_KEYS
        if leaked:
            errors.append(f"{relative}: query keys leaked into memory: {sorted(leaked)}")
        semantic_events = semantic.get("events", [])
        semantic_event_total += len(semantic_events)
        if [event.get("peak_frame") for event in semantic_events] != frames:
            errors.append(f"{relative}: detection/semantic event frames differ")
        for event in semantic_events:
            episodic = event.get("episodic_memory") or {}
            writer = episodic.get("semantic_writer") or {}
            if writer.get("query_conditioned") is not False:
                errors.append(
                    f"{relative}: event {event.get('event_id')} is not query-independent"
                )
            if int(episodic.get("semantic_schema_version", 1)) >= 2:
                required_keys = ("what", "where", "scene", "background", "objects")
            else:
                required_keys = ("what", "where", "anomaly_entity", "anomaly_count")
            for key in required_keys:
                if episodic.get(key) is None:
                    errors.append(
                        f"{relative}: event {event.get('event_id')} missing {key}"
                    )
            if not semantic_memory_complete(episodic):
                errors.append(
                    f"{relative}: event {event.get('event_id')} has invalid or "
                    "contaminated semantic content"
                )

    qa_summary = None
    if qa_path.is_file() and not args.skip_qa:
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        current_memory_digest = memory_set_digest(semantic_root)
        qa_memory_digest = qa.get("memory_digest")
        qa_summary = {
            "samples": int(qa.get("samples", 0)),
            "correct": int(qa.get("correct", 0)),
            "accuracy": float(qa.get("accuracy", 0.0)),
            "model": qa.get("model"),
            "memory_digest": qa_memory_digest,
            "current_memory_digest": current_memory_digest,
            "stale": qa_memory_digest != current_memory_digest,
        }
        if qa_summary["samples"] != len(detection_paths):
            errors.append(
                f"QA samples {qa_summary['samples']} != memories {len(detection_paths)}"
            )
        if qa_summary["stale"]:
            errors.append("QA results were produced from a different semantic memory set")
    elif args.require_qa:
        errors.append(f"Missing QA results: {qa_path}")

    report = {
        "run_root": str(args.run_root),
        "valid": not errors,
        "videos": len(detection_paths),
        "detection_events": event_total,
        "semantic_events": semantic_event_total,
        "qa": qa_summary,
        "errors": errors,
    }
    output_path = args.output_path or args.run_root / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
