from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EventSpan:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float
    frame_count: int


def group_threshold_events(
    frame_rows: list[dict[str, Any]],
    threshold: float,
    merge_gap: int = -1,
    top_fraction: float = 0.05,
    min_patch_iou: float = 0.10,
    max_events: int | None = None,
) -> list[EventSpan]:
    """Create threshold events; negative merge_gap keeps every crossing separate."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if not 0 <= min_patch_iou <= 1:
        raise ValueError("min_patch_iou must be in [0, 1]")

    def active_patches(row: dict[str, Any]) -> set[int]:
        scores = row.get("patch_scores")
        if scores is None:
            return set()
        tensor = torch.as_tensor(scores).reshape(-1)
        keep = max(1, round(tensor.numel() * top_fraction))
        return set(torch.topk(tensor.float(), keep).indices.tolist())

    def patch_iou(left: set[int], right: set[int]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    candidates = sorted(
        (
            (
                int(row["frame_idx"]),
                float(row["event_score"]),
                active_patches(row),
            )
            for row in frame_rows
            if float(row["event_score"]) >= threshold
        ),
        key=lambda item: item[0],
    )
    groups: list[list[tuple[int, float, set[int]]]] = []
    for item in candidates:
        should_merge = bool(groups) and merge_gap >= 0
        if should_merge:
            previous = groups[-1][-1]
            should_merge = (
                item[0] - previous[0] <= merge_gap + 1
                and patch_iou(item[2], previous[2]) >= min_patch_iou
            )
        if not should_merge:
            groups.append([item])
        else:
            groups[-1].append(item)

    events = []
    for group in groups:
        peak_frame, peak_score, _ = max(group, key=lambda item: item[1])
        events.append(
            EventSpan(
                start_frame=group[0][0],
                end_frame=group[-1][0],
                peak_frame=peak_frame,
                peak_score=peak_score,
                frame_count=len(group),
            )
        )
    if max_events is not None and len(events) > max_events:
        events = sorted(events, key=lambda event: event.peak_score, reverse=True)[
            :max_events
        ]
        events.sort(key=lambda event: event.peak_frame)
    return events
