#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from caption_event_memory_qwen25 import (
    describe_physical_where,
    resolve_evidence_path,
    setup_distributed,
)
from vsrtask.semantic_memory import semantic_memory_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite only the where field of existing Qwen2.5 event memory. "
            "All other semantic fields are preserved exactly."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device, rank, world_size = setup_distributed(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    memory_paths = sorted(args.memory_root.rglob("*.json"))[rank::world_size]
    if not memory_paths:
        raise FileNotFoundError(f"No memory JSON files under {args.memory_root}")

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    model.generation_config.temperature = None

    rewritten = 0
    preserved = 0
    for position, memory_path in enumerate(memory_paths, start=1):
        relative = memory_path.relative_to(args.memory_root)
        output_path = args.output_root / relative
        if output_path.is_file() and not args.overwrite:
            print(f"[where-skip] rank={rank} {relative}", flush=True)
            continue
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        for event in memory.get("events", []):
            episodic = event.get("episodic_memory") or {}
            original_path = resolve_evidence_path(
                memory_path, event["evidence"]["original_image"]
            )
            with Image.open(original_path) as source:
                original_image = source.convert("RGB")
            where, attempts = describe_physical_where(
                model,
                processor,
                original_image,
                device,
                str(episodic.get("what") or "the salient foreground object"),
                args.max_new_tokens,
                event.get("evidence", {}).get("selected_patch_indices"),
                event.get("evidence", {}).get("token_grid"),
            )
            raw = episodic.setdefault("raw_response", {})
            raw["physical_where_attempts"] = attempts
            raw["physical_where_source"] = "isolated_rewrite_preserving_other_fields"
            validation = episodic.setdefault("semantic_validation", {})
            validation["mode"] = "isolated_physical_where_v1"
            if where is None:
                fallbacks = validation.setdefault("fallbacks", [])
                fallback = "physical_where_generation_failed_preserve_original"
                if fallback not in fallbacks:
                    fallbacks.append(fallback)
                preserved += 1
            else:
                episodic["where"] = where
                rewritten += 1
            validation["complete"] = semantic_memory_complete(episodic)
            if not validation["complete"]:
                raise RuntimeError(
                    f"Rewritten semantic event is invalid: {relative} "
                    f"frame={event.get('peak_frame')}"
                )

        memory["where_rewriter"] = {
            "model": str(args.model),
            "query_conditioned": False,
            "policy": "isolated_physical_where_v1",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(memory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[where-memory] rank={rank} {position}/{len(memory_paths)} {relative}",
            flush=True,
        )

    shard = args.output_root / f"where_rewrite_summary.rank{rank:03d}.json"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        json.dumps(
            {"rank": rank, "rewritten": rewritten, "preserved": preserved},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if world_size > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        summaries = [
            json.loads(
                (args.output_root / f"where_rewrite_summary.rank{index:03d}.json")
                .read_text(encoding="utf-8")
            )
            for index in range(world_size)
        ]
        manifest = {
            "model": str(args.model),
            "source_memory_root": str(args.memory_root),
            "output_memory_root": str(args.output_root),
            "policy": "isolated_physical_where_v1",
            "rewritten": sum(item["rewritten"] for item in summaries),
            "preserved": sum(item["preserved"] for item in summaries),
        }
        (args.output_root.parent / "where_rewrite_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[where-complete] {json.dumps(manifest)}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
