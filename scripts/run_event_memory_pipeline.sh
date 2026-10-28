#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-all}
if [[ "$STAGE" != "all" && "$STAGE" != "detect" && "$STAGE" != "caption" && "$STAGE" != "qa" ]]; then
    echo "Usage: $0 [all|detect|caption|qa] [extra stage arguments...]" >&2
    exit 2
fi
shift || true

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
DETECTOR_MODEL=${DETECTOR_MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
CAPTION_QA_MODEL=${CAPTION_QA_MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt}
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall}
SOURCE_PARQUET=${SOURCE_PARQUET:-$DATASET_ROOT/test_10mins.parquet}
RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_event_memory_pipeline}
DETECTION_ROOT=${DETECTION_ROOT:-$RUN_ROOT/detection}
SEMANTIC_ROOT=${SEMANTIC_ROOT:-$RUN_ROOT/semantic}
QA_ROOT=${QA_ROOT:-$RUN_ROOT/qa}
VSR_PYTHON=${VSR_PYTHON:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
QWEN35_PYTHON=${QWEN35_PYTHON:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}
THRESHOLD_CONFIG=${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_recall90.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}
THRESHOLD=$(
    "$VSR_PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
        "$THRESHOLD_CONFIG"
)

RANGE_ARGS=()
if [[ -n "${SAMPLE_START:-}" || -n "${SAMPLE_END:-}" ]]; then
    if [[ -z "${SAMPLE_START:-}" || -z "${SAMPLE_END:-}" ]]; then
        echo "SAMPLE_START and SAMPLE_END must be set together" >&2
        exit 2
    fi
    RANGE_ARGS=(--sample-range "$SAMPLE_START" "$SAMPLE_END")
fi

MAX_EVENT_ARGS=()
if [[ -n "${MAX_MEMORY_EVENTS:-}" ]]; then
    MAX_EVENT_ARGS=(--max-memory-events "$MAX_MEMORY_EVENTS")
fi

run_detect() {
    echo "[pipeline:detect] model=$DETECTOR_MODEL checkpoint=$CHECKPOINT"
    "$VSR_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node "$NUM_PROCESSES" \
        "$VSR_ROOT/scripts/build_vsi_event_memory.py" \
        --model "$DETECTOR_MODEL" \
        --checkpoint "$CHECKPOINT" \
        --dataset-root "$DATASET_ROOT" \
        --source-parquet "$SOURCE_PARQUET" \
        --output-root "$DETECTION_ROOT" \
        --threshold "$THRESHOLD" \
        --merge-gap -1 \
        --top-fraction "${TOP_FRACTION:-0.05}" \
        --layer-fusion "${LAYER_FUSION:-mean}" \
        --dtype "${DTYPE:-bfloat16}" \
        "${MAX_EVENT_ARGS[@]}" "${RANGE_ARGS[@]}" "$@"
}

run_caption() {
    test -d "$DETECTION_ROOT/memories" || {
        echo "Missing detection memories: $DETECTION_ROOT/memories" >&2
        exit 2
    }
    echo "[pipeline:caption] model=$CAPTION_QA_MODEL"
    "$QWEN35_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node "$NUM_PROCESSES" \
        "$VSR_ROOT/scripts/caption_event_memory_qwen35.py" \
        --model "$CAPTION_QA_MODEL" \
        --memory-root "$DETECTION_ROOT/memories" \
        --output-root "$SEMANTIC_ROOT/memories" \
        --device auto \
        --dtype "${DTYPE:-bfloat16}" \
        "$@"
}

run_qa() {
    test -d "$SEMANTIC_ROOT/memories" || {
        echo "Missing semantic memories: $SEMANTIC_ROOT/memories" >&2
        exit 2
    }
    echo "[pipeline:qa] model=$CAPTION_QA_MODEL"
    "$QWEN35_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node "$NUM_PROCESSES" \
        "$VSR_ROOT/scripts/evaluate_event_memory_qwen35.py" \
        --model "$CAPTION_QA_MODEL" \
        --memory-root "$SEMANTIC_ROOT/memories" \
        --source-parquet "$SOURCE_PARQUET" \
        --output-dir "$QA_ROOT" \
        --dtype "${DTYPE:-bfloat16}" \
        "${RANGE_ARGS[@]}" "$@"
}

cd "$VSR_ROOT"
case "$STAGE" in
    detect) run_detect "$@" ;;
    caption) run_caption "$@" ;;
    qa) run_qa "$@" ;;
    all)
        run_detect
        run_caption
        run_qa
        "$VSR_PYTHON" "$VSR_ROOT/scripts/validate_event_memory_pipeline.py" \
            --run-root "$RUN_ROOT" --require-qa
        ;;
esac
