#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vsrtask.semantic_memory import (
    repair_semantic_memory,
    semantic_memory_complete,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair streamed Qwen2.5 semantic event files from preserved raw "
            "attempts before falling back to model regeneration."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    event_root = args.run_root / "semantic" / "events"
    event_paths = sorted(event_root.rglob("*.json"))
    if not event_paths:
        raise FileNotFoundError(f"No semantic event files under {event_root}")

    repaired_count = 0
    unresolved = []
    for path in event_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        event = payload.get("event") or {}
        memory = event.get("episodic_memory")
        if semantic_memory_complete(memory):
            continue
        repaired = repair_semantic_memory(memory)
        if repaired is None:
            unresolved.append(str(path))
            continue
        event["episodic_memory"] = repaired
        atomic_json(path, payload)
        repaired_count += 1
        print(
            f"[semantic-repaired-local] video={payload.get('video_path')} "
            f"frame={event.get('peak_frame')} what={repaired.get('what')!r}",
            flush=True,
        )

    print(
        f"[semantic-repair-summary] files={len(event_paths)} "
        f"repaired={repaired_count} unresolved={len(unresolved)}",
        flush=True,
    )
    for path in unresolved:
        print(f"[semantic-repair-needs-model] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
