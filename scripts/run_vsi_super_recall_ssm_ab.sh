#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-both}
DURATION=${DURATION:-10mins}

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
SSM_LAUNCHER=$LMMS_ROOT/scripts/vsr_qwen25ssm.sh
ORIGINAL_ROOT=/data1/ZhangHuayu/datasets/VSI-SUPER-Recall
MASKED_ROOT=${MASKED_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_checkpoint10_top5}
OUTPUT_BASE=${OUTPUT_BASE:-$VSR_ROOT/outputs/vsi_super_recall_ssm_ab_checkpoint10_top5}
TASK=cambrians_vsr_local_${DURATION}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}

run_one() {
    local name=$1
    local video_root=$2
    local port=$3
    VSI_SUPER_RECALL_ROOT="$video_root" \
    TASK="$TASK" \
    MAIN_PROCESS_PORT="$port" \
    OUTPUT_PATH="$OUTPUT_BASE/$DURATION/$name" \
    bash "$SSM_LAUNCHER"
}

validate_masked_dataset() {
    local parquet="$MASKED_ROOT/test_${DURATION}.parquet"
    test -f "$parquet" || {
        echo "Missing predicted-mask dataset: $parquet" >&2
        exit 2
    }
    "$PYTHON_BIN" - "$parquet" "$MASKED_ROOT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

parquet = Path(sys.argv[1])
root = Path(sys.argv[2])
table = pd.read_parquet(parquet)
missing = [str(path) for path in table["video_path"] if not (root / path).is_file()]
if len(table) != 60 or missing:
    print(
        f"Incomplete masked dataset: rows={len(table)}/60 "
        f"missing_videos={len(missing)}",
        file=sys.stderr,
    )
    if missing:
        print("First missing videos:", *missing[:10], sep="\n  ", file=sys.stderr)
    raise SystemExit(2)
print(f"Validated masked dataset: rows={len(table)} videos=60")
PY
}

case "$MODE" in
    original)
        run_one original "$ORIGINAL_ROOT" "${MAIN_PROCESS_PORT:-29731}"
        ;;
    masked)
        validate_masked_dataset
        run_one masked_top5 "$MASKED_ROOT" "${MAIN_PROCESS_PORT:-29732}"
        ;;
    both)
        validate_masked_dataset
        run_one original "$ORIGINAL_ROOT" "${MAIN_PROCESS_PORT_ORIGINAL:-29731}"
        run_one masked_top5 "$MASKED_ROOT" "${MAIN_PROCESS_PORT_MASKED:-29732}"
        ;;
    *)
        echo "Usage: $0 [original|masked|both]" >&2
        exit 2
        ;;
esac
