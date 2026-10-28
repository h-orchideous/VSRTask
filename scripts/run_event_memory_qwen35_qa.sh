#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
MEMORY_ROOT=${MEMORY_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_event_memory_qwen35/memories}
SOURCE_PARQUET=${SOURCE_PARQUET:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall/test_10mins.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$VSR_ROOT/outputs/vsi_super_recall_event_memory_qwen35_qa}
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"

IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}

cd "$VSR_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    scripts/evaluate_event_memory_qwen35.py \
    --model "$MODEL" \
    --memory-root "$MEMORY_ROOT" \
    --source-parquet "$SOURCE_PARQUET" \
    --output-dir "$OUTPUT_DIR" \
    --dtype "${DTYPE:-bfloat16}" \
    "$@"
