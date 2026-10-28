#!/usr/bin/env bash
set -euo pipefail

DURATION=${DURATION:-10mins}
VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
TOP4_ROOT=${TOP4_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_patchdiff_top4_raw}
OUTPUT_PATH=${OUTPUT_PATH:-$VSR_ROOT/outputs/vsi_super_recall_patchdiff_top4_raw_eval/$DURATION}
TASK=cambrians_vsr_local_${DURATION}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}

PARQUET=$TOP4_ROOT/test_${DURATION}.parquet
test -f "$PARQUET" || {
    echo "Missing Top-4 dataset: $PARQUET" >&2
    exit 2
}

"$PYTHON_BIN" - "$PARQUET" "$TOP4_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

parquet = Path(sys.argv[1])
root = Path(sys.argv[2])
table = pd.read_parquet(parquet)
missing = [str(path) for path in table["video_path"] if not (root / path).is_file()]
if len(table) != 60 or missing:
    raise SystemExit(
        f"Incomplete Top-4 dataset: rows={len(table)}/60 missing={len(missing)}"
    )
print("Validated Top-4 raw dataset: rows=60 videos=60")
PY

VSI_SUPER_RECALL_ROOT="$TOP4_ROOT" \
TASK="$TASK" \
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29761} \
NUM_FRAMES=4 \
VIDEO_FPS=1 \
OUTPUT_PATH="$OUTPUT_PATH" \
bash "$LMMS_ROOT/scripts/vsr_qwen25ssm.sh"
