#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from caption_event_memory_qwen25 import describe_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume threshold event jobs and immediately write Qwen2.5 memory."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--producer-count", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


def setup_distributed() -> tuple[torch.device, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    return torch.device("cuda", local_rank), rank, world_size


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def semantic_output(run_root: Path, job: dict[str, Any]) -> Path:
    return run_root / "semantic" / "events" / job["semantic_event_path"]


def recover_queue(run_root: Path) -> None:
    queue = run_root / "queue"
    pending = queue / "pending"
    processing = queue / "processing"
    done = queue / "done"
    for directory in (pending, processing, done, queue / "producers"):
        directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(processing.glob("*.json")):
        os.replace(path, pending / path.name)
    for path in sorted(done.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not semantic_output(run_root, job).is_file():
            os.replace(path, pending / path.name)


def claim_one(run_root: Path) -> Path | None:
    pending = run_root / "queue" / "pending"
    processing = run_root / "queue" / "processing"
    for source in sorted(pending.glob("*.json")):
        claimed = processing / source.name
        try:
            os.replace(source, claimed)
            return claimed
        except FileNotFoundError:
            continue
    return None


def producers_finished(run_root: Path, expected: int) -> bool:
    return len(list((run_root / "queue" / "producers").glob("rank_*.done.json"))) >= expected


def queue_empty(run_root: Path) -> bool:
    queue = run_root / "queue"
    return not any((queue / "pending").glob("*.json")) and not any(
        (queue / "processing").glob("*.json")
    )


def process_job(
    claimed: Path,
    run_root: Path,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
) -> float:
    started_at = time.perf_counter()
    job = json.loads(claimed.read_text(encoding="utf-8"))
    output_path = semantic_output(run_root, job)
    event = job["event"]
    if not output_path.is_file():
        image_path = Path(event["evidence"]["original_image"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        description = describe_event(
            model,
            processor,
            image,
            device,
            args.max_new_tokens,
        )
        description["when"] = {
            "frame": event["peak_frame"],
            "seconds": event["peak_seconds"],
        }
        description["semantic_writer"] = {
            "model": str(args.model),
            "query_conditioned": False,
            "streaming": True,
        }
        event["episodic_memory"] = description
        atomic_json(
            output_path,
            {
                "schema_version": 1,
                "video_path": job["video_path"],
                "event": event,
            },
        )
        print(
            f"[memory-written] rank={rank} video={job['video_path']} "
            f"frame={event['peak_frame']} latency_s={time.perf_counter() - started_at:.2f} "
            f"what={description.get('what')!r}",
            flush=True,
        )
    done_path = run_root / "queue" / "done" / claimed.name
    os.replace(claimed, done_path)
    return time.perf_counter() - started_at


def main() -> int:
    args = parse_args()
    if args.producer_count <= 0:
        raise ValueError("--producer-count must be positive")
    device, rank, world_size = setup_distributed()
    if rank == 0:
        recover_queue(args.run_root)
    if world_size > 1:
        dist.barrier(device_ids=[device.index])

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
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

    completed = 0
    while True:
        claimed = claim_one(args.run_root)
        if claimed is not None:
            process_job(claimed, args.run_root, model, processor, device, args, rank)
            completed += 1
            continue
        if producers_finished(args.run_root, args.producer_count) and queue_empty(args.run_root):
            break
        time.sleep(args.poll_seconds)

    print(f"[consumer-done] rank={rank} events={completed}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
