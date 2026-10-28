#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import site
import subprocess
import sys
import time
from functools import lru_cache
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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.anomaly_detector import TemporalPatchAnomalyDetector, diff_patch_magnitude
from vsrtask.detector_data import load_diff_records, make_windows_for_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one causal sliding-window patch detector. "
            "Each edited frame selects one positive window and clean windows "
            "are sampled as negative windows."
        )
    )
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument(
        "--kv-cache-root",
        type=Path,
        required=True,
        help="Per-sample precomputed K/V feature cache.",
    )
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load detector weights and continue from a saved checkpoint. "
            "The checkpoint epoch is treated as the number of completed epochs."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Total causal window length: current frame plus previous frames.",
    )
    parser.add_argument(
        "--negative-windows",
        type=int,
        default=4,
        help="Clean negative windows sampled per video.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Windows per GPU optimizer step. The global batch is batch-size * world-size.",
    )
    parser.add_argument("--detector-d-model", type=int, default=256)
    parser.add_argument("--detector-layers", type=int, default=2)
    parser.add_argument("--detector-heads", type=int, default=8)
    parser.add_argument(
        "--attention-batch-chunk-size",
        type=int,
        default=4096,
        help=(
            "Maximum number of layer/patch temporal streams passed to the "
            "Transformer attention in one CUDA call."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-range", nargs=2, type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many samples on rank 0.",
    )
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    return device, rank, world_size, distributed


def configure_transformer_attention() -> None:
    """Avoid SDPA CUDA launch limits for the large flattened patch batch.

    The temporal Transformer reshapes B * 28 * 391 patch streams into one
    attention batch. Flash/memory-efficient SDPA can fail with
    ``invalid configuration argument`` at larger detector batch sizes.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def is_main_process(rank: int) -> bool:
    return rank == 0


def json_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def normalize_sample_id(value: Any) -> str:
    text = str(value)
    return f"{int(text):08d}" if text.isdigit() else text


def feature_cache_key(path: Path) -> str:
    return str(path.resolve())


def load_feature_cache(
    cache_path: Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    # Cache files contain several gigabytes of frame tensors, while each
    # training window only touches a small subset. mmap keeps the storages
    # file-backed so unused frames are not read into RAM.
    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Invalid KV cache, missing frames list: {cache_path}")
    features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid frame item in KV cache: {cache_path}")
        path = Path(str(item["path"]))
        k = item["k"]
        v = item["v"]
        if not torch.is_tensor(k) or not torch.is_tensor(v) or k.shape != v.shape:
            raise ValueError(f"Invalid K/V tensors in KV cache: {cache_path}")
        features[feature_cache_key(path)] = (k, v)
    return features, payload


def load_video_map(manifest_path: Path, dataset_root: Path) -> dict[str, Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"Manifest must contain a samples list: {manifest_path}")
    video_map: dict[str, Path] = {}
    for sample in samples:
        if not isinstance(sample, dict) or "sample_id" not in sample:
            continue
        sample_id = normalize_sample_id(sample["sample_id"])
        video_path = Path(str(sample.get("video_path", "")))
        if not video_path.is_absolute():
            video_path = dataset_root / video_path
        video_map[sample_id] = video_path
    return video_map


def load_source_video_map(manifest_path: Path) -> dict[str, Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"Manifest must contain a samples list: {manifest_path}")
    video_map: dict[str, Path] = {}
    for sample in samples:
        if not isinstance(sample, dict) or "sample_id" not in sample:
            continue
        source_video = sample.get("source_video")
        if source_video:
            video_map[normalize_sample_id(sample["sample_id"])] = Path(
                str(source_video)
            )
    return video_map


@lru_cache(maxsize=None)
def probe_video_frame_count(video_path: str) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value.isdigit() or int(value) <= 0:
        raise RuntimeError(
            f"ffprobe could not determine a positive frame count for {video_path}; "
            f"output={value!r}"
        )
    return int(value)


def resolve_clean_video(
    source_video: Path,
    edited_video: Path | None,
) -> Path:
    """Choose a clean video whose frame numbering matches the edited video."""
    candidates = [source_video]
    if source_video.parent.name == "train_merged5":
        candidates.insert(
            0,
            source_video.parent.parent
            / "train_merged5_1fps"
            / source_video.name,
        )
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"Neither source video nor 1 FPS source exists: {source_video}"
        )

    edited_count = None
    if edited_video is not None and edited_video.exists():
        edited_count = probe_video_frame_count(str(edited_video))
    if edited_count is not None:
        for candidate in candidates:
            if probe_video_frame_count(str(candidate)) == edited_count:
                return candidate
    return candidates[0]


def recover_one_edited_frame(
    video_path: Path,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Decode one zero-based frame from a generated video.

    ffmpeg still decodes internally up to the requested frame, but only one
    PNG is materialized. This avoids creating a full temporary frame dump.
    """
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing generated video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}.png"
    )
    if temporary.exists():
        temporary.unlink()
    filter_expr = f"select=eq(n\\,{frame_idx})"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.exists():
            raise FileNotFoundError(
                f"ffmpeg did not produce frame {frame_idx} from {video_path}"
            )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_clean_frame(
    video_path: Path,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Decode one clean source-video frame only when a window needs it."""
    if output_path.exists():
        return
    recover_one_edited_frame(video_path, frame_idx, output_path)


def ensure_edited_frames(
    record: dict,
    video_map: dict[str, Path],
) -> int:
    """Lazily recover only missing edited PNGs referenced by one record."""
    sample_id = normalize_sample_id(record["sample_id"])
    video_path = video_map.get(sample_id)
    if video_path is None:
        raise KeyError(
            f"Sample {sample_id} is not present in the manifest; "
            "cannot recover missing edited frames"
        )
    recovered = 0
    for entry in record.get("entries", []):
        edited_path = Path(str(entry["edited_frame"]))
        if edited_path.exists():
            continue
        recover_one_edited_frame(
            video_path=video_path,
            frame_idx=int(entry["frame_idx"]),
            output_path=edited_path,
        )
        recovered += 1
    return recovered


def main() -> int:
    args = parse_args()
    device, rank, world_size, distributed = setup_distributed(args)
    configure_transformer_attention()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    detector_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if args.window_size <= 0:
        raise ValueError("window-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.attention_batch_chunk_size <= 0:
        raise ValueError("attention-batch-chunk-size must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    window_size = args.window_size

    paths = load_diff_records(
        args.kv_diff_root,
        tuple(args.sample_range) if args.sample_range else None,
        args.max_samples,
    )
    first_sample_id = normalize_sample_id(paths[0].stem)
    first_cache_path = args.kv_cache_root / f"{first_sample_id}.pt"
    first_features, first_cache = load_feature_cache(first_cache_path)
    if args.epochs > int(first_cache.get("cache_epochs", 0)):
        raise ValueError(
            f"KV cache {first_cache_path} was built for "
            f"{first_cache.get('cache_epochs', 0)} epoch(s), but training asks "
            f"for {args.epochs}. Re-run cache_vsr_kv_features with "
            f"--cache-epochs {args.epochs}."
        )
    num_qwen_layers = int(first_cache["num_qwen_layers"])
    kv_dim = int(first_cache["kv_dim"])

    detector = TemporalPatchAnomalyDetector(
        num_qwen_layers=num_qwen_layers,
        kv_dim=kv_dim,
        d_model=args.detector_d_model,
        num_layers=args.detector_layers,
        num_heads=args.detector_heads,
        max_window=window_size,
        attention_batch_chunk_size=args.attention_batch_chunk_size,
    ).to(device=device, dtype=detector_dtype)
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        detector.load_state_dict(resume_payload["detector"])
    if distributed:
        detector = DDP(
            detector,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    if is_main_process(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    history = [] if is_main_process(rank) else None
    epoch_metrics = [] if is_main_process(rank) else None
    start_epoch = 0
    step = 0
    if resume_payload is not None:
        start_epoch = int(resume_payload.get("epoch", 0))
        step = int(resume_payload.get("step", 0))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint already completed {start_epoch} epochs, "
                f"but --epochs={args.epochs}."
            )
        if is_main_process(rank):
            epoch_metrics = list(resume_payload.get("epoch_metrics", []))
            print(
                f"[resume] checkpoint={args.resume_checkpoint} "
                f"completed_epochs={start_epoch} optimizer_steps={step} "
                f"optimizer_state={'loaded' if 'optimizer' in resume_payload else 'not_saved'}",
                flush=True,
            )
    detector_for_summary = detector.module if distributed else detector
    detector_total, detector_trainable = parameter_count(detector_for_summary)
    run_config = json_args(args)
    model_summary = {
        "task": "temporal_patch_diff_regression",
        "backbone": {
            "name": "Qwen2.5-VL-7B-Instruct (precomputed K/V)",
            "frozen": True,
            "features_precomputed": True,
            "layers": num_qwen_layers,
            "kv_feature_dim": kv_dim,
        },
        "detector": {
            "class": "TemporalPatchAnomalyDetector",
            "architecture": "independent_k_v_branches",
            "trainable": True,
            "total_parameters": detector_total,
            "trainable_parameters": detector_trainable,
            "window_size": window_size,
            "history_frames": window_size - 1,
            "visual_tokens": 391,
            "input_shapes": {
                "k": ["batch", window_size, "qwen_layers", "patches", "kv_dim"],
                "v": ["batch", window_size, "qwen_layers", "patches", "kv_dim"],
            },
            "output_shape": [
                "batch",
                "qwen_layers",
                "patches",
                "kv_feature_dim",
            ],
            "output_k": "predicted_delta_K",
            "output_v": "predicted_delta_V",
            "detector_layers": args.detector_layers,
            "detector_hidden_size": args.detector_d_model,
            "detector_heads": args.detector_heads,
            "attention_batch_chunk_size": args.attention_batch_chunk_size,
            "layer_fusion": "diagnostic RMS over predicted K/V deltas, then mean/max",
            "temporal_processing": "one temporal Transformer sequence per layer and patch",
        },
        "data": {
            "positive_window": "previous frames + current edited frame",
            "negative_window": "previous clean frames + current clean frame",
            "negative_windows_per_video": args.negative_windows,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "positive_target": {
                "k": "complete delta_K tensor [qwen_layers, patches, kv_feature_dim]",
                "v": "complete delta_V tensor [qwen_layers, patches, kv_feature_dim]",
            },
            "negative_target": "two all-zero [qwen_layers, patches, kv_feature_dim] delta tensors",
            "loss": "0.5 * MSE(predicted_delta_K, target_delta_K) + 0.5 * MSE(predicted_delta_V, target_delta_V)",
            "supervised_timestep": "last/current frame only",
        },
        "feature_cache": {
            "root": str(args.kv_cache_root),
            "storage_dtype": first_cache.get("dtype"),
            "precomputed_frames_first_sample": len(first_features),
            "cache_epochs": first_cache.get("cache_epochs"),
        },
        "distributed": {
            "world_size": world_size,
            "rank_zero_saves": True,
        },
    }
    if is_main_process(rank):
        (args.output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "model_summary.json").write_text(
            json.dumps(model_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if is_main_process(rank):
        print(
            f"[train_patch_detector] world_size={world_size} "
            f"device={device} distributed={distributed}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        local_loss_sum = 0.0
        local_window_count = 0
        shuffled = paths[:]
        random.Random(args.seed + epoch).shuffle(shuffled)
        local_paths = shuffled[rank::world_size]
        local_sample_count = len(local_paths)
        completed_samples = 0
        progress_windows = 0
        epoch_start = time.monotonic()
        if is_main_process(rank):
            print(
                f"[progress] epoch={epoch + 1}/{args.epochs} "
                f"samples=0/{local_sample_count} windows=0 "
                f"optimizer_steps={step}",
                flush=True,
            )
        join_context = detector.join() if distributed else contextlib.nullcontext()
        with join_context:
            for record_path in local_paths:
                if is_main_process(rank):
                    next_sample_number = completed_samples + 1
                    print(
                        f"[progress] epoch={epoch + 1}/{args.epochs} "
                        f"starting_sample={next_sample_number}/{local_sample_count} "
                        f"id={normalize_sample_id(record_path.stem)}",
                        flush=True,
                    )
                record = torch.load(record_path, map_location="cpu", weights_only=False)
                sample_id = normalize_sample_id(record["sample_id"])
                cache_path = args.kv_cache_root / f"{sample_id}.pt"
                feature_cache, cache_payload = load_feature_cache(cache_path)
                if epoch >= int(cache_payload.get("cache_epochs", 0)):
                    raise ValueError(
                        f"Cache {cache_path} does not contain epoch {epoch}; "
                        "rebuild it with a larger --cache-epochs."
                    )
                frame_count = int(cache_payload["frame_count"])
                windows = make_windows_for_sample(
                    record,
                    args.frames_root,
                    window_size,
                    args.negative_windows,
                    args.seed + epoch,
                    frame_count=frame_count,
                )
                pending = []

                def train_batch(batch_windows: list) -> None:
                    nonlocal step, local_loss_sum
                    nonlocal local_window_count, progress_windows
                    batch_k = []
                    batch_v = []
                    batch_target_k = []
                    batch_target_v = []
                    for batch_window in batch_windows:
                        features_k = []
                        features_v = []
                        for frame_path in batch_window.frame_paths:
                            cached = feature_cache.get(feature_cache_key(frame_path))
                            if cached is None:
                                raise KeyError(
                                    f"Frame {frame_path} is missing from {cache_path}. "
                                    "Re-run cache_vsr_kv_features with matching "
                                    "window/negative/cache-epochs settings."
                                )
                            frame_k, frame_v = cached
                            features_k.append(frame_k)
                            features_v.append(frame_v)
                        batch_k.append(torch.stack(features_k))
                        batch_v.append(torch.stack(features_v))
                        batch_target_k.append(batch_window.target_k)
                        batch_target_v.append(batch_window.target_v)

                    k = torch.stack(batch_k).to(
                        device=device, dtype=detector_dtype
                    )
                    v = torch.stack(batch_v).to(
                        device=device, dtype=detector_dtype
                    )
                    target_k = torch.stack(batch_target_k).to(device)
                    target_v = torch.stack(batch_target_v).to(device)

                    prediction_k, prediction_v = detector(k, v)
                    if (
                        prediction_k.shape != target_k.shape
                        or prediction_v.shape != target_v.shape
                    ):
                        raise RuntimeError(
                            f"Detector output/target mismatch: "
                            f"K {tuple(prediction_k.shape)} vs {tuple(target_k.shape)}, "
                            f"V {tuple(prediction_v.shape)} vs {tuple(target_v.shape)}"
                        )
                    item_loss_k = (
                        prediction_k.float() - target_k.float()
                    ).square().flatten(1).mean(dim=1)
                    item_loss_v = (
                        prediction_v.float() - target_v.float()
                    ).square().flatten(1).mean(dim=1)
                    item_loss = 0.5 * (item_loss_k + item_loss_v)
                    loss = item_loss.mean()
                    prediction = diff_patch_magnitude(prediction_k, prediction_v)
                    target = diff_patch_magnitude(target_k, target_v)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    local_loss_sum += float(item_loss.detach().sum().cpu())
                    local_window_count += len(batch_windows)
                    progress_windows += len(batch_windows)

                    if is_main_process(rank):
                        for index, batch_window in enumerate(batch_windows):
                            row = {
                                "epoch": epoch,
                                "step": step,
                                "sample_id": batch_window.sample_id,
                                "current_frame_idx": batch_window.current_frame_idx,
                                "window_type": (
                                    "edited" if batch_window.positive else "clean"
                                ),
                                "target_diff_mean": float(
                                    target[index].mean().detach().cpu()
                                ),
                                "predicted_diff_mean": float(
                                    prediction[index].mean().detach().cpu()
                                ),
                                "predicted_k_mean": float(
                                    prediction_k[index].mean().detach().cpu()
                                ),
                                "predicted_v_mean": float(
                                    prediction_v[index].mean().detach().cpu()
                                ),
                                "target_k_mean": float(
                                    target_k[index].mean().detach().cpu()
                                ),
                                "target_v_mean": float(
                                    target_v[index].mean().detach().cpu()
                                ),
                                "loss_k": float(item_loss_k[index].detach().cpu()),
                                "loss_v": float(item_loss_v[index].detach().cpu()),
                                "loss": float(item_loss[index].detach().cpu()),
                                "rank": rank,
                                "local_step": step,
                            }
                            history.append(row)
                            if len(history) % args.log_every == 0:
                                print(
                                    json.dumps(row, ensure_ascii=False),
                                    flush=True,
                                )

                for window in windows:
                    pending.append(window)
                    if len(pending) == args.batch_size:
                        train_batch(pending)
                        pending = []
                if pending:
                    train_batch(pending)
                completed_samples += 1
                if (
                    is_main_process(rank)
                    and (
                        completed_samples % args.progress_every == 0
                        or completed_samples == local_sample_count
                    )
                ):
                    elapsed = time.monotonic() - epoch_start
                    samples_per_second = completed_samples / max(elapsed, 1e-6)
                    remaining_samples = local_sample_count - completed_samples
                    eta_seconds = remaining_samples / max(samples_per_second, 1e-6)
                    eta_minutes = eta_seconds / 60.0
                    print(
                        f"[progress] epoch={epoch + 1}/{args.epochs} "
                        f"samples={completed_samples}/{local_sample_count} "
                        f"({100.0 * completed_samples / max(local_sample_count, 1):.1f}%) "
                        f"windows={progress_windows} "
                        f"optimizer_steps={step} "
                        f"speed={samples_per_second:.2f} samples/s "
                        f"eta={eta_minutes:.1f} min "
                        f"current={sample_id}",
                        flush=True,
                    )

        if distributed:
            stats = torch.tensor(
                [local_loss_sum, float(local_window_count)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss_sum = float(stats[0].item())
            global_window_count = int(stats[1].item())
            dist.barrier()
        else:
            global_loss_sum = local_loss_sum
            global_window_count = local_window_count
        if is_main_process(rank):
            epoch_elapsed = time.monotonic() - epoch_start
            print(
                f"[progress] epoch={epoch + 1}/{args.epochs} done "
                f"samples={local_sample_count}/{local_sample_count} "
                f"windows={global_window_count} "
                f"elapsed={epoch_elapsed / 60.0:.1f} min",
                flush=True,
            )
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "global_windows": global_window_count,
                    "global_mean_loss": (
                        global_loss_sum / global_window_count
                        if global_window_count
                        else None
                    ),
                }
            )
            model_state = detector.module.state_dict() if distributed else detector.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "detector": model_state,
                    "optimizer": optimizer.state_dict(),
                    "config": vars(args),
                    "run_config": run_config,
                    "model_summary": model_summary,
                    "epoch_metrics": epoch_metrics,
                    "world_size": world_size,
                },
                args.output_dir / f"checkpoint-{epoch + 1}.pt",
            )
            (args.output_dir / "train_log.json").write_text(
                json.dumps(
                    {
                        "window_logs": history,
                        "epoch_metrics": epoch_metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
