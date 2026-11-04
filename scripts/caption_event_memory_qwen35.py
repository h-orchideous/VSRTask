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
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from vsrtask.semantic_memory import (
    OVERLAY_ARTIFACT_REJECT,
    SEMANTIC_TEXT_REJECT,
    build_structured_event_memory,
    choose_anomaly_description,
    choose_placement_text,
    entity_from_what,
    entity_matches_what,
    full_frame_observation_complete,
    merge_observation_attempts,
    parse_anomaly_response,
    parse_full_frame_observation,
    parse_physical_where_response,
    recover_unstructured_observation,
    semantic_memory_complete,
)
from vsrtask.semantic_evidence import build_detector_focus_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen3.5 to describe detector-selected event images without "
            "exposing the downstream VSR question."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--memory-root", type=Path)
    parser.add_argument(
        "--overlay-image",
        type=Path,
        help=(
            "PatchDiff audit evidence paired with --image in smoke-test mode. "
            "It is recorded but never passed to the semantic model."
        ),
    )
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
        count_value = int(count)
        if (
            normalized
            and len(re.sub(r"[^a-z0-9]", "", normalized)) >= 2
            and count_value > 0
            and not OVERLAY_ARTIFACT_REJECT.search(normalized)
        ):
            counts.append({"name": normalized, "count": count_value})
    return counts


def generate(
    model: torch.nn.Module,
    processor: Any,
    images: list[Image.Image],
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
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
            use_cache=True,
        )
    generated = output[:, batch["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def describe_physical_where(
    model: torch.nn.Module,
    processor: Any,
    original_image: Image.Image,
    device: torch.device,
    what: str,
    max_new_tokens: int,
    selected_patch_indices: list[int] | None = None,
    token_grid: list[int] | tuple[int, int] | None = None,
) -> tuple[str | None, list[str]]:
    focus_crop = build_detector_focus_crop(
        original_image, selected_patch_indices, token_grid
    )
    images = [original_image, focus_crop] if focus_crop is not None else [original_image]
    prompts = [
        (
            f"Target object or event: {what}. State only its visible physical "
            "placement relative to the nearest support, container, furniture, "
            "or surface. Examples: inside the trash can; on the bed; beside the "
            "desk. Do not mention image coordinates, the frame, a crop, or a "
            "detector. Answer with one concise location phrase only."
        ),
        (
            f"Where is {what} physically placed? Answer only with a relation and "
            "visible anchor, such as 'on the chair' or 'inside the basket'. Never "
            "answer with left, right, center, top, or bottom of the image."
        ),
    ]
    attempts = []
    for prompt in prompts:
        response = generate(
            model,
            processor,
            images,
            prompt,
            device,
            min(max_new_tokens, 32),
        )
        attempts.append(response)
        if where := parse_physical_where_response(response):
            return where, attempts
    return None, attempts


def describe_anomaly(
    model: torch.nn.Module,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str | None, list[str]]:
    prompts = [
        (
            "Name the single most unusual foreground physical object in these "
            "source images. The second image, when present, is a clean focus crop. "
            "Return only one noun phrase of at most 10 words. Include distinctive "
            "identity or appearance, but no location, scene, labels, or explanation."
        ),
        (
            "Identify the salient physical object in the focused region. Return "
            "only its short object name or identity, with no prefix or punctuation."
        ),
    ]
    attempts = []
    for prompt in prompts:
        response = generate(
            model,
            processor,
            images,
            prompt,
            device,
            min(max_new_tokens, 32),
        )
        attempts.append(response)
        if anomaly := parse_anomaly_response(response):
            return anomaly, attempts
    return None, attempts


def describe_event(
    model: torch.nn.Module,
    processor: Any,
    original_image: Image.Image,
    overlay_image: Image.Image | None,
    device: torch.device,
    max_new_tokens: int,
    selected_patch_indices: list[int] | None = None,
    token_grid: list[int] | tuple[int, int] | None = None,
) -> dict[str, Any]:
    fallbacks: list[str] = []
    focus_crop = build_detector_focus_crop(
        original_image,
        selected_patch_indices,
        token_grid,
    )
    observation_images = [original_image]
    focus_instruction = ""
    if focus_crop is not None:
        observation_images.append(focus_crop)
        focus_instruction = (
            " The second image is an unmodified crop from the first image, "
            "centered on the detector's highest-scoring visual token. Use it "
            "only to inspect that source-image region."
        )
    count_images = [original_image]
    observation_prompt = (
        "Describe this entire video frame for question-independent event memory."
        + focus_instruction
        + " Identify the most visually unusual or salient foreground object or "
        "event, then describe the full visible scene. Do not describe image "
        "annotations, crop boundaries, or detector visualizations. Do not infer "
        "an unseen question, cause, or temporal change. State only visible facts. "
        "Output exactly one line with these four labels, each once. Keep every "
        "value under 12 words. What: name and briefly describe the unusual "
        "foreground object. Where: start with a physical relation and name the "
        "nearest visible support, container, furniture, or surface. Scene: name "
        "the overall scene type. Background: list other visible surroundings."
    )
    observation = generate(
        model,
        processor,
        observation_images,
        observation_prompt,
        device,
        min(max_new_tokens, 128),
    )
    fields = parse_full_frame_observation(observation)
    candidate = fields
    observation_attempts = [observation]
    if not full_frame_observation_complete(candidate):
        observation = generate(
            model,
            processor,
            observation_images,
            (
                "Return a clean observation based only on visible source-image "
                "content. Do not describe annotations or crop boundaries. Do not "
                "explain the task and do not use brackets, markup, "
                "or template text. Output exactly one line using the labels What, "
                "Where, Scene, and Background, with concise visible content after "
                "each label. The What value must name a physical object, and the "
                "Where value must start with a physical relation such as on, in, "
                "inside, beside, or under. Keep each value under 12 words. Do not "
                "emit angle brackets, option letters, or placeholder text."
            ),
            device,
            min(max_new_tokens, 128),
        )
        observation_attempts.append(observation)
        fields = parse_full_frame_observation(observation)
        candidate = fields
    if not full_frame_observation_complete(candidate):
        observation = generate(
            model,
            processor,
            observation_images,
            (
                "Inspect the source image again. Return exactly four short values "
                "on one line separated by || and do not add labels: unusual "
                "foreground object || physical relation plus anchor || scene type "
                "|| other visible surroundings. Do not output option letters, "
                "instructions, markup, or placeholders."
            ),
            device,
            min(max_new_tokens, 128),
        )
        observation_attempts.append(observation)
        fields = parse_full_frame_observation(observation)
        candidate = fields
    if not full_frame_observation_complete(candidate):
        for attempt in reversed(observation_attempts):
            recovered = recover_unstructured_observation(attempt)
            if recovered is not None:
                fields = recovered
                candidate = fields
                fallbacks.append("fields_recovered_from_unstructured_observation")
                break
    anomaly_description, anomaly_attempts = describe_anomaly(
        model,
        processor,
        observation_images,
        device,
        max_new_tokens,
    )
    if anomaly_description is None:
        anomaly_description = parse_anomaly_response(candidate.get("what"))
        if anomaly_description is None:
            raise RuntimeError(
                "Qwen did not produce a valid anomaly phrase and the full-frame "
                f"observation was not usable: {anomaly_attempts!r}"
            )
        fallbacks.append("anomaly_from_full_frame_observation")
    if not full_frame_observation_complete(candidate):
        merged = merge_observation_attempts(
            observation_attempts,
            fallback_what=anomaly_description,
        )
        if merged is not None:
            fields, missing_fields = merged
            candidate = fields
            fallbacks.extend(
                f"{field}_unresolved" for field in missing_fields
            )
            fallbacks.append("fields_merged_across_image_attempts")
    fields = dict(candidate)
    fields["what"] = choose_anomaly_description(
        anomaly_description,
        fields.get("what"),
    )
    candidate = fields
    if not full_frame_observation_complete(candidate):
        raise RuntimeError(
            "Qwen did not produce a valid full-frame observation: "
            f"{observation!r}"
        )

    count_prompt = (
        "Inventory visible physical objects in the full original frame. The "
        "salient candidate from the observation is: "
        f"{fields['what']}. Output the anomaly first, then up to seven other "
        "salient visible object types. Output only semicolon-separated lowercase "
        "name=count pairs, for example: cat=1; chair=1; desk=1. Do not explain."
    )
    count_response = generate(
        model,
        processor,
        count_images,
        count_prompt,
        device,
        min(max_new_tokens, 64),
    )
    counts = parse_counts(count_response)
    count_attempts = [count_response]
    if (
        not counts
        or SEMANTIC_TEXT_REJECT.search(count_response)
        or OVERLAY_ARTIFACT_REJECT.search(count_response)
    ):
        count_response = generate(
            model,
            processor,
            count_images,
            (
                f"The salient visible object is {fields['what']}. Output only "
                "unique lowercase object=count pairs from the full frame, separated "
                "by semicolons. Put the anomaly first. Maximum 8 pairs."
            ),
            device,
            min(max_new_tokens, 64),
        )
        count_attempts.append(count_response)
        counts = parse_counts(count_response)

    expected_entity = entity_from_what(fields["what"])
    matching_index = next(
        (
            index
            for index, item in enumerate(counts)
            if entity_matches_what(str(item["name"]), expected_entity)
        ),
        None,
    )
    if matching_index is None:
        anomaly = {"name": expected_entity, "count": 1}
        counts.insert(0, anomaly)
        fallbacks.append("anomaly_from_observation_count_one")
    else:
        anomaly = counts.pop(matching_index)
        counts.insert(0, anomaly)

    targeted_where, targeted_where_attempts = describe_physical_where(
        model,
        processor,
        original_image,
        device,
        fields["what"],
        max_new_tokens,
        selected_patch_indices,
        token_grid,
    )
    selected_where = choose_placement_text(
        fields["where"],
        anomaly_attempts[-1],
        targeted_where,
    )
    structured = build_structured_event_memory(
        fields,
        counts,
        targeted_where=selected_where,
    )
    placement_status = structured["placement"]["status"]
    if placement_status == "targeted_retry":
        fallbacks.append("placement_targeted_retry")
    elif placement_status == "unresolved":
        fallbacks.append("placement_unresolved")

    result = {
        "semantic_schema_version": 3,
        "what": fields["what"],
        "where": fields["where"],
        "scene": fields["scene"],
        "background": fields["background"],
        "objects": counts,
        **structured,
        "raw_response": {
            "full_frame_observation": observation,
            "counts": count_response,
            "observation_attempts": observation_attempts,
            "count_attempts": count_attempts,
            "anomaly_attempts": anomaly_attempts,
            "targeted_where_attempts": targeted_where_attempts,
            "targeted_where": targeted_where,
            "selected_where": selected_where,
            "input_images": (
                ["original", "detector_focus_crop"]
                if focus_crop is not None
                else ["original"]
            ),
            "overlay_evidence_available": overlay_image is not None,
            "semantic_input_policy": "clean_source_pixels_only",
        },
        "semantic_validation": {
            "complete": True,
            "fallbacks": fallbacks,
            "mode": "v3_anomaly_placement_context",
        },
    }
    if not semantic_memory_complete(result):
        raise RuntimeError(f"Full-frame semantic memory validation failed: {result}")
    return result


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
    if output_path.is_file() and not args.overwrite:
        try:
            existing_output = json.loads(output_path.read_text(encoding="utf-8"))
            existing_events = existing_output.get("events", [])
            if len(existing_events) == len(events) and all(
                semantic_memory_complete(event.get("episodic_memory"))
                for event in existing_events
            ):
                print(f"[skip-valid] {output_path}", flush=True)
                return 0
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
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
        overlay_value = event.get("evidence", {}).get("overlay_image")
        overlay_path = (
            resolve_evidence_path(memory_path, overlay_value)
            if overlay_value
            else None
        )
        with Image.open(image_path) as source:
            original_image = source.convert("RGB")
        if overlay_path is not None:
            with Image.open(overlay_path) as source:
                overlay_image = source.convert("RGB")
        else:
            overlay_image = None
        description = describe_event(
            model,
            processor,
            original_image,
            overlay_image,
            device,
            args.max_new_tokens,
            event.get("evidence", {}).get("selected_patch_indices"),
            event.get("evidence", {}).get("token_grid"),
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
    relationships["count_source"] = "qwen3.5_event_observation"
    relationships["object_count_evolution"] = [
        {
            "event_id": event.get("event_id"),
            "frame": event.get("peak_frame"),
            "seconds": event.get("peak_seconds"),
            "counts": (event.get("episodic_memory") or {}).get("objects")
            or (event.get("episodic_memory") or {})
            .get("dynamic_scene_relationships", {})
            .get("object_counts", []),
        }
        for event in events
    ]
    relationships.pop("location_relationship_evolution", None)
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
    if args.overlay_image and not args.image:
        raise ValueError("--overlay-image can only be used with --image")
    if args.memory_root and not args.output_root:
        raise ValueError("--output-root is required with --memory-root")

    device, rank, world_size = setup_distributed(args.device)
    if args.image and world_size > 1:
        raise ValueError("--image smoke testing only supports one process")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    if args.image:
        with Image.open(args.image) as source:
            original_image = source.convert("RGB")
        if args.overlay_image:
            with Image.open(args.overlay_image) as source:
                overlay_image = source.convert("RGB")
        else:
            overlay_image = None
        result = describe_event(
            model,
            processor,
            original_image,
            overlay_image,
            device,
            args.max_new_tokens,
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
