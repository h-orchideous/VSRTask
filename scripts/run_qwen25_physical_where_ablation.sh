#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen25_10mins_v3_t0888_compact}
SOURCE_MEMORY_ROOT=${SOURCE_MEMORY_ROOT:-$RUN_ROOT/semantic/memories}
SEMANTIC_MEMORY_ROOT=${SEMANTIC_MEMORY_ROOT:-$RUN_ROOT/semantic_physical_where_isolated/memories}
SOURCE_PARQUET=${SOURCE_PARQUET:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall/test_10mins.parquet}
QA_OUTPUT_DIR=${QA_OUTPUT_DIR:-$RUN_ROOT/qa_ablation_physical_where_isolated}
GPUS=${GPUS:-4,5,6,7}

IFS=',' read -r -a DEVICE_LIST <<< "$GPUS"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICE_LIST[@]}}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"

cd "$VSR_ROOT"
echo "[physical-where] rewriting only where on GPUs $GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" \
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    "$VSR_ROOT/scripts/rewrite_where_memory_qwen25.py" \
    --model "$MODEL" \
    --memory-root "$SOURCE_MEMORY_ROOT" \
    --output-root "$SEMANTIC_MEMORY_ROOT" \
    --device auto \
    --dtype "${DTYPE:-bfloat16}" \
    --overwrite

echo "[physical-where] evaluating QA from isolated where rewrite"
CUDA_VISIBLE_DEVICES="$GPUS" \
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    "$VSR_ROOT/scripts/evaluate_event_memory_qwen25.py" \
    --model "$MODEL" \
    --memory-root "$SEMANTIC_MEMORY_ROOT" \
    --source-parquet "$SOURCE_PARQUET" \
    --output-dir "$QA_OUTPUT_DIR" \
    --event-retrieval all \
    --dtype "${DTYPE:-bfloat16}"
