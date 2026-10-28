#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import site
import sys
from pathlib import Path

# A stale user-site DeepSpeed can be incompatible with the PyTorch version in
# the Qwen environment. Keep this standalone task isolated from ~/.local.
user_site = getattr(site, "USER_SITE", None)
if user_site:
    sys.path = [
        path for path in sys.path
        if Path(path).resolve() != Path(user_site).resolve()
    ]

import torch
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsrtask.diff_transformer import KDiffTransformer, VDiffTransformer
from vsrtask.qwen_kv import capture_edited_kv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the only active VSR task: separate K/V Transformer "
            "predictors from edited-frame and clean-frame windows."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kv-diff-root", type=Path, required=True)
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Clean extracted frames: <frames-root>/<sample-id>/<frame+1:06d>.png",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2, help="Transformer block count in each small network.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Causal window length; only the last frame is supervised.",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--negative-frames-per-sample",
        type=int,
        default=16,
        help="Number of clean zero-target windows sampled from each video.",
    )
    parser.add_argument(
        "--sample-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive numeric sample-id range.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def load_record_paths(
    root: Path,
    max_samples: int | None,
    sample_range: tuple[int, int] | None,
) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*.pt")):
        sample_id = path.stem
        if sample_range is not None and sample_id.isdigit():
            number = int(sample_id)
            if not sample_range[0] <= number <= sample_range[1]:
                continue
        paths.append(path)
        if max_samples is not None and len(paths) >= max_samples:
            break
    if not paths:
        raise ValueError(f"No .pt diff records found under {root}")
    return paths


def target_tensor(entry: dict, key: str, device: torch.device, max_layers: int | None) -> torch.Tensor:
    values = entry[key]
    indices = sorted(values, key=lambda x: int(x))
    if max_layers is not None:
        indices = indices[:max_layers]
    # Stored tensors are [1, T, D]; training uses [L, T, D].
    return torch.stack([values[index].squeeze(0) for index in indices]).to(device)


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    kwargs = dict(
        input_dim=512,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_window=args.window_size,
    )
    return KDiffTransformer(**kwargs).to(device), VDiffTransformer(**kwargs).to(device)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if args.sample_range and args.sample_range[0] > args.sample_range[1]:
        raise ValueError("--sample-range START must be <= END")
    if args.negative_frames_per_sample < 0:
        raise ValueError("--negative-frames-per-sample must be >= 0")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    record_paths = load_record_paths(
        args.kv_diff_root,
        args.max_samples,
        tuple(args.sample_range) if args.sample_range else None,
    )
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    qwen.eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)

    k_net, v_net = build_models(args, device)
    print(
        "[train_diff_transformer] task=edited_frame_kv_delta_prediction "
        "trainable=KDiffTransformer,VDiffTransformer "
        f"window={args.window_size} "
        f"negative_windows_per_sample={args.negative_frames_per_sample}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        list(k_net.parameters()) + list(v_net.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0

    for epoch in range(args.epochs):
        random.shuffle(record_paths)
        for record_path in record_paths:
            # Keep only one sample's targets in RAM. The complete target
            # directory is tens of gigabytes.
            record = torch.load(record_path, map_location="cpu", weights_only=False)
            entries = record.get("entries", [])
            if args.max_entries is not None:
                entries = entries[:args.max_entries]
            if not entries:
                raise ValueError(f"Diff record has no entries: {record_path}")

            sample_id = str(record["sample_id"])
            edited_indices = {int(entry["frame_idx"]) for entry in entries}
            clean_dir = args.frames_root / sample_id
            clean_candidates = {
                int(path.stem) - 1: path
                for path in clean_dir.glob("*.png")
                if path.stem.isdigit()
            }
            if args.negative_frames_per_sample and not clean_candidates:
                raise FileNotFoundError(
                    f"No clean frames found under {clean_dir}; "
                    "clean frames are required for zero-target negatives"
                )
            def window_indices(current: int) -> list[int]:
                start = max(0, current - args.window_size + 1)
                indices = list(range(start, current + 1))
                # Repeat the first available frame when the video does not
                # have enough history near its beginning.
                while len(indices) < args.window_size:
                    indices.insert(0, indices[0])
                return indices

            negative_indices = [
                index for index in sorted(clean_candidates)
                if index not in edited_indices
                and not (set(window_indices(index)) & edited_indices)
            ]
            random.shuffle(negative_indices)
            negative_indices = negative_indices[:args.negative_frames_per_sample]
            if (
                args.negative_frames_per_sample
                and len(negative_indices) < args.negative_frames_per_sample
            ):
                raise ValueError(
                    f"{sample_id} has only {len(negative_indices)} available clean "
                    f"frames, requested {args.negative_frames_per_sample}"
                )

            edited_paths = {
                int(entry["frame_idx"]): Path(str(entry["edited_frame"]))
                for entry in entries
            }

            def make_window(current: int, use_edited_current: bool) -> list[Path]:
                paths = []
                for index in window_indices(current):
                    if use_edited_current and index in edited_paths:
                        path = edited_paths[index]
                    else:
                        path = clean_candidates.get(index)
                        if path is None:
                            raise FileNotFoundError(
                                f"Missing clean frame for {sample_id} frame_idx={index}"
                            )
                    if not path.exists():
                        raise FileNotFoundError(f"Missing window frame: {path}")
                    paths.append(path)
                return paths

            examples: list[
                tuple[str, int, list[Path], torch.Tensor, torch.Tensor]
            ] = []
            for entry in entries:
                edited_frame = entry.get("edited_frame")
                if not edited_frame or not Path(edited_frame).exists():
                    raise FileNotFoundError(f"Missing edited frame: {edited_frame}")
                current = int(entry["frame_idx"])
                target_k = target_tensor(entry, "delta_k", device, args.max_layers)
                target_v = target_tensor(entry, "delta_v", device, args.max_layers)
                examples.append(
                    (
                        "edited",
                        current,
                        make_window(current, use_edited_current=True),
                        target_k,
                        target_v,
                    )
                )

            for frame_idx in negative_indices:
                window = make_window(frame_idx, use_edited_current=False)
                examples.append(
                    (
                        "clean",
                        frame_idx,
                        window,
                        torch.zeros_like(examples[0][3], device=device),
                        torch.zeros_like(examples[0][4], device=device),
                    )
                )
            random.shuffle(examples)

            for example_type, frame_idx, window, target_k, target_v in examples:
                # Keep feature memory bounded to one 8-frame window. Windows
                # overlap, but retaining every frame from a whole video can
                # unnecessarily consume several GB of host memory.
                window_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
                window_k = []
                window_v = []
                for frame_path in window:
                    if frame_path not in window_cache:
                        window_cache[frame_path] = capture_edited_kv(
                            qwen, processor, frame_path, device, dtype
                        )
                    frame_k, frame_v = window_cache[frame_path]
                    if args.max_layers is not None:
                        frame_k = frame_k[:args.max_layers]
                        frame_v = frame_v[:args.max_layers]
                    window_k.append(frame_k)
                    window_v.append(frame_v)
                window_k = torch.stack(window_k).to(
                    device=device, dtype=torch.float32
                )
                window_v = torch.stack(window_v).to(
                    device=device, dtype=torch.float32
                )
                prediction_k = k_net(window_k)
                prediction_v = v_net(window_v)
                if prediction_k.shape != target_k.shape or prediction_v.shape != target_v.shape:
                    raise RuntimeError(
                        f"Input/target shape mismatch for {sample_id} "
                        f"frame_idx={frame_idx}: "
                        f"K {tuple(prediction_k.shape)} vs {tuple(target_k.shape)}, "
                        f"V {tuple(prediction_v.shape)} vs {tuple(target_v.shape)}"
                    )
                loss_k = loss_fn(prediction_k, target_k.float())
                loss_v = loss_fn(prediction_v, target_v.float())
                loss = loss_k + loss_v
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1

                row = {
                    "epoch": epoch,
                    "step": step,
                    "sample_id": sample_id,
                    "frame_idx": frame_idx,
                    "example_type": example_type,
                    "window_paths": [str(path) for path in window],
                    "loss": float(loss.detach().cpu()),
                    "loss_k": float(loss_k.detach().cpu()),
                    "loss_v": float(loss_v.detach().cpu()),
                }
                history.append(row)
                if step % args.log_every == 0:
                    print(json.dumps(row, ensure_ascii=False), flush=True)

        checkpoint = {
            "epoch": epoch + 1,
            "step": step,
            "k_diff_net": k_net.state_dict(),
            "v_diff_net": v_net.state_dict(),
            "config": vars(args),
        }
        torch.save(checkpoint, args.output_dir / f"checkpoint-{epoch + 1}.pt")
        (args.output_dir / "train_log.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
