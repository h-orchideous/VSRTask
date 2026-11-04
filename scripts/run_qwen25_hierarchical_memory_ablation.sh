#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen25_10mins_v3_t0888_compact}
MEMORY_ROOT=${MEMORY_ROOT:-$RUN_ROOT/semantic/memories}
SOURCE_PARQUET=${SOURCE_PARQUET:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall/test_10mins.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/qa_ablation_vsr_hierarchical_old_threshold}
OLD_THRESHOLD=${OLD_THRESHOLD:-0.9073798060417175}
GPUS=${GPUS:-4,5,6,7}

IFS=',' read -r -a DEVICE_LIST <<< "$GPUS"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICE_LIST[@]}}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"

cd "$VSR_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" \
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_PROCESSES" \
    "$VSR_ROOT/scripts/evaluate_event_memory_qwen25.py" \
    --model "$MODEL" \
    --memory-root "$MEMORY_ROOT" \
    --source-parquet "$SOURCE_PARQUET" \
    --output-dir "$OUTPUT_DIR" \
    --event-score-threshold "$OLD_THRESHOLD" \
    --event-retrieval all \
    --memory-view vsr_hierarchical \
    --dtype "${DTYPE:-bfloat16}"
