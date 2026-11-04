from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image


def build_detector_focus_crop(
    image: Image.Image,
    selected_patch_indices: Sequence[int] | None,
    token_grid: Sequence[int] | None,
    *,
    radius_cells: int = 2,
) -> Image.Image | None:
    """Crop clean source pixels around the detector's highest-scoring token."""
    if not selected_patch_indices or not token_grid or len(token_grid) != 2:
        return None
    rows, cols = (int(token_grid[0]), int(token_grid[1]))
    if rows <= 0 or cols <= 0 or radius_cells < 0:
        return None

    index = int(selected_patch_indices[0])
    if index < 0 or index >= rows * cols:
        return None
    row, col = divmod(index, cols)

    left_col = max(0, col - radius_cells)
    right_col = min(cols, col + radius_cells + 1)
    top_row = max(0, row - radius_cells)
    bottom_row = min(rows, row + radius_cells + 1)
    width, height = image.size
    box = (
        math.floor(left_col * width / cols),
        math.floor(top_row * height / rows),
        math.ceil(right_col * width / cols),
        math.ceil(bottom_row * height / rows),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.crop(box)
