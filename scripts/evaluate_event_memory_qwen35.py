#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from vsrtask.qa_event_selection import EventSelection, select_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VSR multiple-choice QA from explicit event memory."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--event-score-threshold",
        type=float,
        help="Evaluate only memory events whose peak_score meets this threshold.",
    )
    parser.add_argument(
        "--event-retrieval",
        choices=("all", "entity"),
        default="all",
        help="Optionally retrieve events matching the entity named in the question.",
    )
    parser.add_argument(
        "--memory-view",
        choices=("full", "vsr_compact", "vsr_hierarchical"),
        default="full",
        help=(
            "full includes every field; vsr_compact exposes only anomaly and "
            "placement; vsr_hierarchical keeps context as an explicit fallback."
        ),
    )
    return parser.parse_args()


def setup_distributed() -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}"), rank, world_size


def sample_id(video_path: str) -> int:
    stem = Path(video_path).stem
    match = re.search(r"(\d{8})$", stem)
    if match is None:
        raise ValueError(f"Cannot extract sample id from {video_path}")
    return int(match.group(1))


def memory_path_for(memory_root: Path, video_path: str) -> Path:
    relative = Path(video_path)
    return memory_root / relative.parent / f"{relative.stem}.json"


def memory_set_digest(memory_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(memory_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No semantic memories under {memory_root}")
    for path in paths:
        digest.update(str(path.relative_to(memory_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def format_counts(memory: dict[str, Any]) -> str:
    entities = memory.get("objects") or memory.get("entities") or []
    return ", ".join(
        f"{item.get('name')}={item.get('count')}" for item in entities
    ) or "none"


def format_memory(events: list[dict[str, Any]], memory_view: str) -> str:
    if not events:
        return "No anomaly events were written."
    lines = []
    for position, event in enumerate(events, start=1):
        episodic = event.get("episodic_memory") or {}
        schema_version = int(episodic.get("semantic_schema_version", 1))
        anomaly = episodic.get("anomaly") or {}
        placement = episodic.get("placement") or {}
        context = episodic.get("context") or {}
        anomaly_text = anomaly.get("entity") or episodic.get("what") or "unknown"
        placement_text = placement.get("text") or episodic.get("where") or "unknown"
        scene_text = context.get("scene") or episodic.get("scene") or "unknown"
        background_text = (
            context.get("summary") or episodic.get("background") or "unknown"
        )
        if memory_view == "vsr_compact":
            lines.append(
                " | ".join(
                    [
                        f"Event {position}",
                        f"anomaly={anomaly_text}",
                        f"placement={placement_text}",
                    ]
                )
            )
            continue
        if memory_view == "vsr_hierarchical":
            lines.extend(
                [
                    f"Event {position}",
                    f"  anomaly: {anomaly_text}",
                    f"  primary_placement: {placement_text}",
                    "  fallback_context: "
                    + "; ".join(
                        [
                            f"scene={scene_text}",
                            f"background={background_text}",
                            f"visible_objects={format_counts(episodic)}",
                        ]
                    ),
                ]
            )
            continue
        if schema_version >= 3:
            lines.append(
                " | ".join(
                    [
                        f"Event {position}",
                        f"anomaly.entity={anomaly_text}",
                        f"anomaly.attributes={anomaly.get('description') or episodic.get('what') or 'unknown'}",
                        f"anomaly.count={anomaly.get('count', 'unknown')}",
                        f"placement={placement_text}",
                        f"placement.status={placement.get('status') or 'unknown'}",
                        f"context.scene={scene_text}",
                        f"context.summary={background_text}",
                    ]
                )
            )
            continue
        if schema_version >= 2:
            lines.append(
                " | ".join(
                    [
                        f"Event {position}",
                        f"what={episodic.get('what') or 'unknown'}",
                        f"where={episodic.get('where') or 'unknown'}",
                        f"scene={episodic.get('scene') or 'unknown'}",
                        f"background={episodic.get('background') or 'unknown'}",
                        f"objects={format_counts(episodic)}",
                    ]
                )
            )
            continue
        lines.append(
            " | ".join(
                [
                    f"Event {position}",
                    f"what={episodic.get('what') or 'unknown'}",
                    f"where={episodic.get('where') or 'unknown'}",
                    f"counts={format_counts(episodic)}",
                ]
            )
        )
    return "\n".join(lines)


def build_prompt(
    row: dict[str, Any], selection: EventSelection, memory_view: str
) -> str:
    options = "\n".join(str(option) for option in row["options"])
    hierarchy_instruction = ""
    if memory_view == "vsr_hierarchical":
        hierarchy_instruction = (
            " For each target event, treat primary_placement as the preferred "
            "location evidence. Consult fallback_context only when that placement "
            "is vague, non-physical, or unknown. Objects merely listed in fallback "
            "context are not automatically the target object's location."
        )
    return (
        "You are answering a video temporal-order multiple-choice question from "
        "an event memory. The memory was created before seeing the question and "
        "is already ordered by video time. Some detected events can be irrelevant "
        "or missing. Match the entity named in the question to the event descriptions, "
        "then compare its chronological locations with the options."
        + hierarchy_instruction
        + "\n\n"
        f"Event memory:\n{format_memory(selection.events, memory_view)}\n\n"
        f"Question:\n{str(row['question']).strip()}\n"
        f"Options:\n{options}\n\n"
        "Return only one option letter: A, B, C, or D."
    )


def generate_answer(
    model: torch.nn.Module,
    processor: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
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


def normalize_choice(response: str) -> str | None:
    match = re.search(r"(?:^|\b)([ABCD])(?:\b|$)", response.upper())
    return match.group(1) if match else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    device, rank, world_size = setup_distributed()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.source_parquet).sort_values("video_path")
    if args.sample_range:
        start, end = args.sample_range
        table = table[
            table["video_path"].map(lambda value: start <= sample_id(value) <= end)
        ]
    rows = table.to_dict("records")
    local_rows = rows[rank::world_size]
    current_memory_digest = memory_set_digest(args.memory_root)

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    results = []
    for position, row in enumerate(local_rows, start=1):
        memory_path = memory_path_for(args.memory_root, row["video_path"])
        if not memory_path.is_file():
            raise FileNotFoundError(f"Missing memory: {memory_path}")
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        selection = select_events(
            memory,
            str(row["question"]),
            score_threshold=args.event_score_threshold,
            retrieval=args.event_retrieval,
        )
        prompt = build_prompt(row, selection, args.memory_view)
        response = generate_answer(
            model, processor, prompt, device, args.max_new_tokens
        )
        prediction = normalize_choice(response)
        target = str(row["answer"]).strip().upper()
        result = {
            "sample_id": sample_id(row["video_path"]),
            "video_path": row["video_path"],
            "memory_path": str(memory_path),
            "event_count": selection.retrieved_event_count,
            "source_event_count": selection.source_event_count,
            "threshold_event_count": selection.threshold_event_count,
            "retrieved_event_count": selection.retrieved_event_count,
            "query_entity": selection.query_entity,
            "retrieval_fallback": selection.retrieval_fallback,
            "memory_view": args.memory_view,
            "question": row["question"],
            "options": [str(option) for option in row["options"]],
            "target": target,
            "raw_response": response,
            "prediction": prediction,
            "correct": prediction == target,
        }
        results.append(result)
        print(
            f"[qa] rank={rank} {position}/{len(local_rows)} "
            f"id={result['sample_id']:08d} events={result['event_count']} "
            f"pred={prediction} target={target} correct={result['correct']}",
            flush=True,
        )

    part_path = args.output_dir / f"predictions.rank{rank:03d}.jsonl"
    write_jsonl(part_path, results)
    if world_size > 1:
        dist.barrier(device_ids=[device.index])

    if rank == 0:
        merged = []
        for part_rank in range(world_size):
            part = args.output_dir / f"predictions.rank{part_rank:03d}.jsonl"
            with part.open("r", encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        merged.sort(key=lambda item: item["sample_id"])
        correct = sum(bool(item["correct"]) for item in merged)
        summary = {
            "model": str(args.model),
            "memory_root": str(args.memory_root),
            "memory_digest": current_memory_digest,
            "source_parquet": str(args.source_parquet),
            "event_selection": {
                "score_threshold": args.event_score_threshold,
                "retrieval": args.event_retrieval,
                "memory_view": args.memory_view,
            },
            "samples": len(merged),
            "correct": correct,
            "accuracy": correct / len(merged) if merged else 0.0,
            "results": merged,
        }
        output_path = args.output_dir / "results.json"
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[complete] correct={correct}/{len(merged)} "
            f"accuracy={summary['accuracy']:.4f} output={output_path}",
            flush=True,
        )

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
