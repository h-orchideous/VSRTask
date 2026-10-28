#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen2.5-VL to describe detector-selected event images without "
            "exposing the downstream VSR question."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--memory-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_counts(text: str) -> list[dict[str, Any]]:
    counts = []
    for name, count in re.findall(r"([^;=:\n{}]+?)\s*[=:]\s*(\d+)", text):
        normalized = name.strip().strip('"\'').lower()
        if normalized:
            counts.append({"name": normalized, "count": int(count)})
    return counts


def parse_narrative(text: str) -> dict[str, Any]:
    fields: dict[str, str | None] = {}
    for key in ("Where", "What", "Relations"):
        match = re.search(
            rf"{key}:\s*(.*?)(?=;\s*(?:Where|What|Relations):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        fields[key.lower()] = match.group(1).strip() if match else None
    return fields


def generate(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.10,
            no_repeat_ngram_size=3,
            use_cache=True,
        )
    generated = output[:, batch["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def is_degenerate(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 20:
        return False
    unique_ratio = len(set(words)) / len(words)
    most_common = max(words.count(word) for word in set(words))
    return unique_ratio < 0.35 or most_common >= 8


def describe_event(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    narrative_prompt = (
        "Describe only what is visibly present in this video frame. Do not infer "
        "a hidden question or an answer. Never repeat words or phrases. Use "
        "exactly this format in at most 35 "
        "words: Where: <scene and spatial context>; What: <most visually unusual "
        "foreground entity or event>; Relations: <its relation to nearby objects>."
    )
    narrative = generate(
        model,
        processor,
        image,
        narrative_prompt,
        device,
        min(max_new_tokens, 64),
    )
    if is_degenerate(narrative):
        narrative = generate(
            model,
            processor,
            image,
            (
                "Identify the single most unusual foreground object in this frame. "
                "Do not repeat any word. Output exactly: Where: <place>; What: "
                "<object>; Relations: <relation to nearby objects>. Use at most "
                "25 words."
            ),
            device,
            min(max_new_tokens, 48),
        )
    count_prompt = (
        "The unusual foreground event was described as follows: "
        f"{narrative} "
        "Count visible objects. The FIRST pair MUST name and count that unusual "
        "foreground entity. Then include up to seven nearby salient object types. "
        "Output only semicolon-separated singular lowercase name=count pairs, "
        "for example: dog=1; sofa=1; table=2. Do not repeat entries, output "
        "template words, or explain."
    )
    count_response = generate(
        model,
        processor,
        image,
        count_prompt,
        device,
        min(max_new_tokens, 48),
    )
    fields = parse_narrative(narrative)
    counts = parse_counts(count_response)
    if not counts or re.search(r"addCriterion|add[A-Z]|<\|", count_response):
        count_response = generate(
            model,
            processor,
            image,
            (
                f"The unusual object is: {fields.get('what') or 'unknown object'}. "
                "Output only unique lowercase object=count pairs separated by "
                "semicolons. Put the unusual object first. Maximum 8 pairs."
            ),
            device,
            min(max_new_tokens, 48),
        )
        counts = parse_counts(count_response)
    anomaly = counts[0] if counts else {"name": None, "count": None}
    relations = [fields["relations"]] if fields["relations"] else []
    return {
        "where": fields["where"],
        "what": fields["what"],
        "narrative": narrative,
        "anomaly_entity": anomaly["name"],
        "anomaly_count": anomaly["count"],
        "nearby_objects": counts[1:],
        "entities": counts,
        "dynamic_scene_relationships": {
            "object_counts": counts,
            "location_relationships": relations,
        },
        "raw_response": {
            "narrative": narrative,
            "counts": count_response,
        },
    }


def resolve_evidence_path(memory_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (memory_path.parent / path).resolve()


def enrich_memory(
    memory_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    events = memory.get("events", [])
    completed = 0
    for event in events:
        if args.max_events is not None and completed >= args.max_events:
            break
        existing = event.get("episodic_memory") or {}
        if existing.get("semantic_writer", {}).get("model") and not args.overwrite:
            continue
        image_path = resolve_evidence_path(
            memory_path, event["evidence"]["original_image"]
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        description = describe_event(
            model,
            processor,
            image,
            device,
            args.max_new_tokens,
        )
        description["when"] = existing.get(
            "when",
            {
                "frame": event.get("peak_frame"),
                "seconds": event.get("peak_seconds"),
            },
        )
        description["semantic_writer"] = {
            "model": str(args.model),
            "query_conditioned": False,
        }
        event["episodic_memory"] = description
        completed += 1
        print(
            f"[caption] memory={memory_path.name} event={event.get('event_id')} "
            f"frame={event.get('peak_frame')} what={description['what']!r}",
            flush=True,
        )

    memory["semantic_writer"] = {
        "model": str(args.model),
        "query_conditioned": False,
        "events_captioned_this_run": completed,
    }
    relationships = memory.setdefault("dynamic_scene_relationships", {})
    relationships["count_source"] = "qwen2.5_vl_event_observation"
    relationships["object_count_evolution"] = [
        {
            "event_id": event.get("event_id"),
            "frame": event.get("peak_frame"),
            "seconds": event.get("peak_seconds"),
            "counts": (event.get("episodic_memory") or {})
            .get("dynamic_scene_relationships", {})
            .get("object_counts", []),
        }
        for event in events
    ]
    relationships["location_relationship_evolution"] = [
        {
            "event_id": event.get("event_id"),
            "frame": event.get("peak_frame"),
            "seconds": event.get("peak_seconds"),
            "relations": (event.get("episodic_memory") or {})
            .get("dynamic_scene_relationships", {})
            .get("location_relationships", []),
        }
        for event in events
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {output_path} captioned={completed}", flush=True)
    return completed


def setup_distributed(device_arg: str) -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device_arg == "auto":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    if world_size > 1:
        if device.type != "cuda":
            raise ValueError("Distributed captioning requires CUDA devices")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
    return device, rank, world_size


def main() -> int:
    args = parse_args()
    if args.image and not args.output_json:
        raise ValueError("--output-json is required with --image")
    if args.memory_root and not args.output_root:
        raise ValueError("--output-root is required with --memory-root")

    device, rank, world_size = setup_distributed(args.device)
    if args.image and world_size > 1:
        raise ValueError("--image smoke testing only supports one process")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        result = describe_event(
            model, processor, image, device, args.max_new_tokens
        )
        result["semantic_writer"] = {
            "model": str(args.model),
            "query_conditioned": False,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        print(f"[saved] {args.output_json}", flush=True)
        return 0

    memory_paths = sorted(args.memory_root.rglob("*.json"))
    if not memory_paths:
        raise FileNotFoundError(f"No memory JSON files under {args.memory_root}")
    all_memory_paths = memory_paths
    memory_paths = all_memory_paths[rank::world_size]
    total = 0
    for position, memory_path in enumerate(memory_paths, start=1):
        relative = memory_path.relative_to(args.memory_root)
        output_path = args.output_root / relative
        print(
            f"[memory] rank={rank} {position}/{len(memory_paths)} {relative}",
            flush=True,
        )
        total += enrich_memory(
            memory_path,
            output_path,
            model,
            processor,
            device,
            args,
        )
    shard_path = args.output_root / f"caption_summary.rank{rank:03d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "memories": len(memory_paths),
                "events_captioned": total,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if world_size > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        shards = [
            json.loads(
                (args.output_root / f"caption_summary.rank{index:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            for index in range(world_size)
        ]
        manifest = {
            "schema_version": 1,
            "model": str(args.model),
            "query_conditioned": False,
            "source_memory_root": str(args.memory_root),
            "output_memory_root": str(args.output_root),
            "memories": sum(item["memories"] for item in shards),
            "events_captioned": sum(item["events_captioned"] for item in shards),
            "world_size": world_size,
        }
        manifest_path = args.output_root.parent / "semantic_memory_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[complete] memories={manifest['memories']} "
            f"events_captioned={manifest['events_captioned']} "
            f"manifest={manifest_path}",
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
