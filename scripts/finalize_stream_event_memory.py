#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble streamed semantic event records into ordered video memories."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    detection_root = args.run_root / "detection" / "memories"
    event_root = args.run_root / "semantic" / "events"
    output_root = args.run_root / "semantic" / "memories"
    detection_paths = sorted(detection_root.rglob("*.json"))
    if not detection_paths:
        raise FileNotFoundError(f"No detection memories under {detection_root}")

    total_events = 0
    model_names: set[str] = set()
    output_paths = []
    for detection_path in detection_paths:
        relative = detection_path.relative_to(detection_root)
        detection = json.loads(detection_path.read_text(encoding="utf-8"))
        events = sorted(detection.get("events", []), key=lambda item: item["peak_frame"])
        semantic_events = []
        for order, event in enumerate(events, start=1):
            event_path = (
                event_root / relative.parent / relative.stem /
                f"frame_{int(event['peak_frame']):06d}.json"
            )
            if not event_path.is_file():
                raise FileNotFoundError(
                    f"Missing streamed semantic event for {relative} "
                    f"frame {event['peak_frame']}: {event_path}"
                )
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            semantic_event = payload["event"]
            if int(semantic_event["peak_frame"]) != int(event["peak_frame"]):
                raise ValueError(f"Frame mismatch in {event_path}")
            semantic_event["event_id"] = order
            semantic_event["temporal_order"] = order
            writer = semantic_event.get("episodic_memory", {}).get("semantic_writer", {})
            if writer.get("model"):
                model_names.add(str(writer["model"]))
            semantic_events.append(semantic_event)

        memory = dict(detection)
        memory["memory_type"] = "streaming_semantic_event_memory"
        memory["event_count"] = len(semantic_events)
        memory["events"] = semantic_events
        memory["semantic_writer"] = {
            "models": sorted(model_names),
            "query_conditioned": False,
            "streaming": True,
        }
        relationships = memory.setdefault("dynamic_scene_relationships", {})
        relationships["count_source"] = "qwen2.5_vl_streaming_event_observation"
        relationships["cross_event_object_tracking"] = False
        relationships["object_count_evolution"] = [
            {
                "event_id": event["event_id"],
                "frame": event["peak_frame"],
                "seconds": event["peak_seconds"],
                "counts": event["episodic_memory"]
                .get("dynamic_scene_relationships", {})
                .get("object_counts", []),
            }
            for event in semantic_events
        ]
        relationships["location_relationship_evolution"] = [
            {
                "event_id": event["event_id"],
                "frame": event["peak_frame"],
                "seconds": event["peak_seconds"],
                "relations": event["episodic_memory"]
                .get("dynamic_scene_relationships", {})
                .get("location_relationships", []),
            }
            for event in semantic_events
        ]
        output_path = output_root / relative
        atomic_json(output_path, memory)
        output_paths.append(output_path)
        total_events += len(semantic_events)
        print(f"[assembled] {relative} events={len(semantic_events)}", flush=True)

    detection_manifest = {
        "schema_version": 1,
        "streaming": True,
        "videos": len(detection_paths),
        "events": sum(
            len(json.loads(path.read_text(encoding="utf-8")).get("events", []))
            for path in detection_paths
        ),
        "memories": [str(path) for path in detection_paths],
    }
    atomic_json(args.run_root / "detection" / "memory_manifest.json", detection_manifest)
    semantic_manifest = {
        "schema_version": 1,
        "streaming": True,
        "query_conditioned": False,
        "models": sorted(model_names),
        "videos": len(output_paths),
        "events": total_events,
        "memories": [str(path) for path in output_paths],
    }
    atomic_json(args.run_root / "semantic" / "semantic_memory_manifest.json", semantic_manifest)
    print(
        f"[complete] videos={len(output_paths)} events={total_events} "
        f"output={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
