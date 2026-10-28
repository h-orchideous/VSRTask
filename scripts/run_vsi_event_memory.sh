#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt}
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall}
SOURCE_PARQUET=${SOURCE_PARQUET:-$DATASET_ROOT/test_10mins.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_event_memory}
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
THRESHOLD_CONFIG=${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_recall90.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"

THRESHOLD=$(
    "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
        "$THRESHOLD_CONFIG"
)
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}
CAPTION_ARGS=()
if [ "${CAPTION_EVENTS:-0}" = "1" ]; then
    CAPTION_ARGS+=(--caption-events)
fi
MAX_EVENT_ARGS=()
if [[ -n "${MAX_MEMORY_EVENTS:-}" ]]; then
    MAX_EVENT_ARGS=(--max-memory-events "$MAX_MEMORY_EVENTS")
fi

cd "$VSR_ROOT"
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    scripts/build_vsi_event_memory.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --source-parquet "$SOURCE_PARQUET" \
    --output-root "$OUTPUT_ROOT" \
    --threshold "$THRESHOLD" \
    --merge-gap "${MERGE_GAP:--1}" \
    --min-patch-iou "${MIN_PATCH_IOU:-0.10}" \
    --top-fraction "${TOP_FRACTION:-0.05}" \
    --layer-fusion "${LAYER_FUSION:-mean}" \
    --dtype "${DTYPE:-bfloat16}" \
    "${MAX_EVENT_ARGS[@]}" \
    "${CAPTION_ARGS[@]}" \
    "$@"
