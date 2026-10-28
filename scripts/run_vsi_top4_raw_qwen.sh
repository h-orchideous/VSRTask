#!/usr/bin/env bash
set -euo pipefail

DURATION=${DURATION:-10mins}
VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
TOP4_ROOT=${TOP4_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_patchdiff_top4_raw}
OUTPUT_PATH=${OUTPUT_PATH:-$VSR_ROOT/outputs/vsi_super_recall_patchdiff_top4_raw_qwen_eval/$DURATION}
TASK=cambrians_vsr_local_${DURATION}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VSI_SUPER_RECALL_ROOT="$TOP4_ROOT"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$LMMS_ROOT:$LMMS_ROOT/..:${PYTHONPATH:-}"

PARQUET=$TOP4_ROOT/test_${DURATION}.parquet
test -x "$PYTHON_BIN" || {
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    exit 2
}
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

IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}
MODEL_ARGS=${MODEL_ARGS:-"pretrained=$CHECKPOINT,min_pixels=100352,max_pixels=100352,max_num_frames=4,fps=1,use_custom_video_loader=True,max_image_size=448,attn_implementation=sdpa,interleave_visuals=False,use_fast_processor=True"}

mkdir -p "$OUTPUT_PATH"
echo "[qwen-top4] task=$TASK samples=60 frames_per_video=4"
echo "[qwen-top4] devices=$CUDA_VISIBLE_DEVICES processes=$NUM_PROCESSES"
echo "[qwen-top4] output=$OUTPUT_PATH"

cd "$LMMS_ROOT"
"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "${MAIN_PROCESS_PORT:-29762}" \
    -m lmms_eval \
    --model qwen2_5_vl \
    --force_simple \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix patchdiff_top4_raw_qwen \
    --output_path "$OUTPUT_PATH"
