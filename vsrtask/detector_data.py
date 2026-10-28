from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

@dataclass
class DetectorWindow:
    sample_id: str
    current_frame_idx: int
    frame_indices: list[int]
    frame_paths: list[Path]
    positive: bool
    target_k: torch.Tensor  # [qwen_layers, visual_tokens, kv_dim]
    target_v: torch.Tensor  # [qwen_layers, visual_tokens, kv_dim]

    @property
    def target(self) -> torch.Tensor:
        """Return a patch-level RMS target for diagnostics."""
        k_score = self.target_k.float().pow(2).mean(dim=-1).sqrt()
        v_score = self.target_v.float().pow(2).mean(dim=-1).sqrt()
        return torch.sqrt(0.5 * (k_score.square() + v_score.square()))


def normalize_sample_id(value: Any) -> str:
    text = str(value)
    return f"{int(text):08d}" if text.isdigit() else text


def _frame_path(frames_root: Path, sample_id: str, frame_idx: int) -> Path:
    return frames_root / sample_id / f"{frame_idx + 1:06d}.png"


def load_diff_records(
    diff_root: Path,
    sample_range: tuple[int, int] | None = None,
    max_samples: int | None = None,
) -> list[Path]:
    paths = []
    for path in sorted(diff_root.glob("*.pt")):
        if sample_range and path.stem.isdigit():
            number = int(path.stem)
            if not sample_range[0] <= number <= sample_range[1]:
                continue
        paths.append(path)
        if max_samples is not None and len(paths) >= max_samples:
            break
    if not paths:
        raise ValueError(f"No diff records found under {diff_root}")
    return paths


def make_windows_for_sample(
    record: dict[str, Any],
    frames_root: Path,
    window_size: int,
    negative_windows: int,
    seed: int,
    frame_count: int | None = None,
    require_edited_frames: bool = True,
) -> list[DetectorWindow]:
    """Build one training set for one video sample.

    Every edited event creates one positive window. Only a small random
    subset of clean windows is added as negatives; the full video is not
    enumerated as training examples.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    sample_id = normalize_sample_id(record["sample_id"])
    entries = sorted(record.get("entries", []), key=lambda x: int(x["frame_idx"]))
    if not entries:
        raise ValueError(f"Sample {sample_id} has no diff entries")

    edited_paths: dict[int, Path] = {}
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry in entries:
        index = int(entry["frame_idx"])
        path = Path(str(entry["edited_frame"]))
        if require_edited_frames and not path.exists():
            raise FileNotFoundError(f"Missing edited frame: {path}")
        edited_paths[index] = path
        delta_k = torch.stack(
            [entry["delta_k"][key] for key in sorted(entry["delta_k"], key=lambda x: int(x))]
        ).squeeze(1)
        delta_v = torch.stack(
            [entry["delta_v"][key] for key in sorted(entry["delta_v"], key=lambda x: int(x))]
        ).squeeze(1)
        # Keep the complete cached delta vectors. The two detector branches
        # directly regress these tensors.
        targets[index] = (delta_k.float(), delta_v.float())

    sample_frames = frames_root / sample_id
    clean_paths = sorted(sample_frames.glob("*.png"))
    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError(
                f"frame_count must be positive for {sample_id}, got {frame_count}"
            )
        # The files are materialized lazily by the training loop. Keeping the
        # expected paths here lets us sample clean negatives without dumping
        # every video frame to disk first.
        available = {
            index: _frame_path(frames_root, sample_id, index)
            for index in range(frame_count)
        }
    else:
        if not clean_paths:
            raise FileNotFoundError(f"No clean frames found under {sample_frames}")
        available = {
            int(path.stem) - 1: path
            for path in clean_paths
            if path.stem.isdigit()
        }
    missing = [idx for idx in edited_paths if idx not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing clean frames for {sample_id}: {missing[:8]}"
        )

    def build_window(current: int, positive: bool) -> DetectorWindow:
        # The current frame is the last frame of a causal window.
        start = max(0, current - window_size + 1)
        indices = list(range(start, current + 1))
        while len(indices) < window_size:
            indices.insert(0, indices[0])
        paths = [
            edited_paths.get(index, available.get(index, _frame_path(frames_root, sample_id, index)))
            for index in indices
        ]
        first_target_k, first_target_v = next(iter(targets.values()))
        zero_target_k = torch.zeros_like(first_target_k)
        zero_target_v = torch.zeros_like(first_target_v)
        target_k, target_v = (
            targets[current] if positive else (zero_target_k, zero_target_v)
        )
        return DetectorWindow(
            sample_id=sample_id,
            current_frame_idx=current,
            frame_indices=indices,
            frame_paths=paths,
            positive=positive,
            target_k=target_k,
            target_v=target_v,
        )

    windows = [
        build_window(int(entry["frame_idx"]), positive=True)
        for entry in entries
    ]
    event_indices = set(edited_paths)
    candidates = [
        index for index in sorted(available)
        if index not in event_indices
        and not any(
            event_index in event_indices
            for event_index in range(max(0, index - window_size + 1), index + 1)
        )
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for index in candidates[:negative_windows]:
        windows.append(build_window(index, positive=False))
    return windows
