#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import site
import sys
import time
from pathlib import Path
from typing import Any

user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.detector_data import load_diff_records, make_windows_for_sample
from vsrtask.kv_feature_cache import (
    feature_cache_key,
    load_feature_cache,
    normalize_sample_id,
)
from vsrtask.patch_diff_detector import (
    TemporalPatchDiffDetector,
    patch_diff_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the standalone temporal patch-diff magnitude detector."
    )
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument("--kv-cache-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--sample-range", nargs=2, type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--negative-windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--detector-d-model", type=int, default=256)
    parser.add_argument("--detector-layers", type=int, default=2)
    parser.add_argument("--detector-heads", type=int, default=8)
    parser.add_argument("--attention-batch-chunk-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device, rank, world_size, distributed


def configure_attention(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def stack_batch(
    windows: list,
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_k = []
    batch_v = []
    targets = []
    for window in windows:
        window_k = []
        window_v = []
        for frame_path in window.frame_paths:
            cached = feature_cache.get(feature_cache_key(frame_path))
            if cached is None:
                raise KeyError(
                    f"Frame {frame_path} is missing from {cache_path}. Rebuild "
                    "the cache with matching window/negative/cache-epochs settings."
                )
            frame_k, frame_v = cached
            window_k.append(frame_k)
            window_v.append(frame_v)
        batch_k.append(torch.stack(window_k))
        batch_v.append(torch.stack(window_v))
        targets.append(patch_diff_target(window.target_k, window.target_v))
    return torch.stack(batch_k), torch.stack(batch_v), torch.stack(targets)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.window_size <= 0:
        raise ValueError("epochs, batch-size, and window-size must be positive")
    if args.negative_windows < 0:
        raise ValueError("negative-windows must be non-negative")

    device, rank, world_size, distributed = setup_distributed(args)
    configure_attention(device)
    torch.manual_seed(args.seed + rank)
    detector_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    paths = load_diff_records(
        args.kv_diff_root,
        tuple(args.sample_range) if args.sample_range else None,
        args.max_samples,
    )
    first_id = normalize_sample_id(paths[0].stem)
    _, first_cache = load_feature_cache(args.kv_cache_root / f"{first_id}.pt")
    cache_epochs = int(first_cache.get("cache_epochs", 1))
    if args.epochs > cache_epochs:
        raise ValueError(
            f"Cache contains negative windows for {cache_epochs} epoch(s), "
            f"but training requested {args.epochs}."
        )

    detector = TemporalPatchDiffDetector(
        num_qwen_layers=int(first_cache["num_qwen_layers"]),
        kv_dim=int(first_cache["kv_dim"]),
        d_model=args.detector_d_model,
        num_layers=args.detector_layers,
        num_heads=args.detector_heads,
        max_window=args.window_size,
        max_visual_tokens=int(first_cache["num_visual_tokens"]),
        attention_batch_chunk_size=args.attention_batch_chunk_size,
    ).to(device=device, dtype=detector_dtype)

    resume = None
    start_epoch = 0
    optimizer_step = 0
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        detector.load_state_dict(resume["detector"], strict=True)
        start_epoch = int(resume.get("epoch", 0))
        optimizer_step = int(resume.get("step", 0))
    if distributed:
        detector = DDP(
            detector,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if resume and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])

    base_detector = detector.module if distributed else detector
    parameter_count = sum(parameter.numel() for parameter in base_detector.parameters())
    run_config = jsonable_args(args)
    model_summary = {
        "task": "temporal_patch_diff_magnitude_regression",
        "backbone": {
            "name": first_cache.get("backbone_name", "Qwen2.5-VL"),
            "model": first_cache.get("backbone_model"),
            "attention_layers": first_cache.get("attention_layers"),
            "frozen": True,
            "layers": base_detector.num_qwen_layers,
            "kv_feature_dim": base_detector.kv_dim,
        },
        "detector": {
            "class": "TemporalPatchDiffDetector",
            "total_parameters": parameter_count,
            "window_size": args.window_size,
            "history_frames": args.window_size - 1,
            "visual_tokens": base_detector.max_visual_tokens,
            "input_shapes": {
                "k": ["batch", args.window_size, "qwen_layers", "patches", "kv_dim"],
                "v": ["batch", args.window_size, "qwen_layers", "patches", "kv_dim"],
            },
            "output_shape": ["batch", "qwen_layers", "patches"],
            "detector_layers": args.detector_layers,
            "detector_hidden_size": args.detector_d_model,
            "detector_heads": args.detector_heads,
            "temporal_processing": "independent temporal sequence per Qwen layer and patch",
        },
        "data": {
            "positive_target": "RMS(delta_K, delta_V) over kv_feature_dim",
            "negative_target": "all-zero [qwen_layers, patches] matrix",
            "loss": "MSE(predicted_patch_diff, target_patch_diff)",
            "negative_windows_per_video": args.negative_windows,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
        },
        "distributed": {"world_size": world_size},
    }
    history = list(resume.get("history", [])) if resume and rank == 0 else []
    epoch_metrics = list(resume.get("epoch_metrics", [])) if resume and rank == 0 else []

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.output_dir / "model_summary.json").write_text(
            json.dumps(model_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[patch_diff] world_size={world_size} parameters={parameter_count:,} "
            f"device={device}",
            flush=True,
        )
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        shuffled = paths[:]
        random.Random(args.seed + epoch).shuffle(shuffled)
        local_paths = shuffled[rank::world_size]
        local_loss_sum = 0.0
        local_window_count = 0
        completed_samples = 0
        epoch_start = time.monotonic()
        if rank == 0:
            print(
                f"[progress] epoch={epoch + 1}/{args.epochs} samples=0/{len(local_paths)}",
                flush=True,
            )

        join_context = detector.join() if distributed else contextlib.nullcontext()
        with join_context:
            for record_path in local_paths:
                record = torch.load(record_path, map_location="cpu", weights_only=False)
                sample_id = normalize_sample_id(record["sample_id"])
                cache_path = args.kv_cache_root / f"{sample_id}.pt"
                feature_cache, cache_payload = load_feature_cache(cache_path)
                if epoch >= int(cache_payload.get("cache_epochs", 1)):
                    raise ValueError(f"Cache {cache_path} does not contain epoch {epoch}")
                windows = make_windows_for_sample(
                    record,
                    args.frames_root,
                    args.window_size,
                    args.negative_windows,
                    args.seed + epoch,
                    frame_count=int(cache_payload["frame_count"]),
                    require_edited_frames=False,
                )

                for start in range(0, len(windows), args.batch_size):
                    batch_windows = windows[start : start + args.batch_size]
                    k, v, target = stack_batch(batch_windows, feature_cache, cache_path)
                    k = k.to(device=device, dtype=detector_dtype)
                    v = v.to(device=device, dtype=detector_dtype)
                    target = target.to(device=device)
                    prediction = detector(k, v)
                    item_loss = (
                        prediction.float() - target.float()
                    ).square().flatten(1).mean(dim=1)
                    loss = item_loss.mean()

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    optimizer_step += 1
                    local_loss_sum += float(item_loss.detach().sum().cpu())
                    local_window_count += len(batch_windows)

                    if rank == 0:
                        for index, window in enumerate(batch_windows):
                            row = {
                                "epoch": epoch,
                                "step": optimizer_step,
                                "sample_id": window.sample_id,
                                "current_frame_idx": window.current_frame_idx,
                                "window_type": "edited" if window.positive else "clean",
                                "target_diff_mean": float(target[index].mean().detach().cpu()),
                                "predicted_diff_mean": float(prediction[index].mean().detach().cpu()),
                                "loss": float(item_loss[index].detach().cpu()),
                            }
                            history.append(row)
                            if len(history) % args.log_every == 0:
                                print(json.dumps(row, ensure_ascii=False), flush=True)

                completed_samples += 1
                if rank == 0 and (
                    completed_samples % args.progress_every == 0
                    or completed_samples == len(local_paths)
                ):
                    elapsed = time.monotonic() - epoch_start
                    rate = completed_samples / max(elapsed, 1e-6)
                    eta = (len(local_paths) - completed_samples) / max(rate, 1e-6) / 60
                    print(
                        f"[progress] epoch={epoch + 1}/{args.epochs} "
                        f"samples={completed_samples}/{len(local_paths)} "
                        f"windows={local_window_count} steps={optimizer_step} "
                        f"eta={eta:.1f} min current={sample_id}",
                        flush=True,
                    )

        stats = torch.tensor(
            [local_loss_sum, float(local_window_count)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_loss_sum = float(stats[0].item())
        global_window_count = int(stats[1].item())
        if rank == 0:
            metric = {
                "epoch": epoch + 1,
                "global_windows": global_window_count,
                "global_mean_loss": global_loss_sum / max(global_window_count, 1),
            }
            epoch_metrics.append(metric)
            state = detector.module.state_dict() if distributed else detector.state_dict()
            payload = {
                "epoch": epoch + 1,
                "step": optimizer_step,
                "detector": state,
                "optimizer": optimizer.state_dict(),
                "run_config": run_config,
                "model_summary": model_summary,
                "epoch_metrics": epoch_metrics,
                "history": history,
                "world_size": world_size,
            }
            torch.save(payload, args.output_dir / f"checkpoint-{epoch + 1}.pt")
            (args.output_dir / "train_log.json").write_text(
                json.dumps(
                    {"epoch_metrics": epoch_metrics, "window_logs": history},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[epoch] {json.dumps(metric)}", flush=True)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
