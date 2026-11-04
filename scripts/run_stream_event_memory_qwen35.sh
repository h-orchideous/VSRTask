#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-stream}
if [[ "$MODE" != "stream" && "$MODE" != "all" && "$MODE" != "finalize" && "$MODE" != "qa" ]]; then
    echo "Usage: $0 [stream|all|finalize|qa]" >&2
    exit 2
fi

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen35_1_377_epoch10_4gpu/checkpoint-10.pt}
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall}
DURATION=${DURATION:-10mins}
case "$DURATION" in
    10mins|30mins|60mins|120mins|240mins) ;;
    *)
        echo "DURATION must be one of: 10mins, 30mins, 60mins, 120mins, 240mins" >&2
        exit 2
        ;;
esac
SOURCE_PARQUET=${SOURCE_PARQUET:-$DATASET_ROOT/test_${DURATION}.parquet}
RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen35_${DURATION}}
QA_MEMORY_ROOT=${QA_MEMORY_ROOT:-$RUN_ROOT/semantic/memories}
QA_OUTPUT_DIR=${QA_OUTPUT_DIR:-$RUN_ROOT/qa}
THRESHOLD_CONFIG=${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_qwen35.json}
DETECT_GPUS=${DETECT_GPUS:-0,1}
SEMANTIC_GPUS=${SEMANTIC_GPUS:-2,3}
DETECT_MASTER_PORT=${DETECT_MASTER_PORT:-29710}
SEMANTIC_MASTER_PORT=${SEMANTIC_MASTER_PORT:-29720}
QA_MASTER_PORT=${QA_MASTER_PORT:-29730}
QA_EVENT_RETRIEVAL=${QA_EVENT_RETRIEVAL:-all}
QA_MEMORY_VIEW=${QA_MEMORY_VIEW:-full}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$SOURCE_PARQUET" ]]; then
    echo "Missing official VSR parquet: $SOURCE_PARQUET" >&2
    exit 2
fi
if [[ ! -f "$THRESHOLD_CONFIG" ]]; then
    echo "Missing Qwen3.5 threshold config: $THRESHOLD_CONFIG" >&2
    echo "Calibrate it with evaluate_temporal_frame_detection_qwen35.py first." >&2
    exit 2
fi

IFS=',' read -r -a DETECT_DEVICE_LIST <<< "$DETECT_GPUS"
IFS=',' read -r -a SEMANTIC_DEVICE_LIST <<< "$SEMANTIC_GPUS"
DETECT_PROCESSES=${DETECT_PROCESSES:-${#DETECT_DEVICE_LIST[@]}}
SEMANTIC_PROCESSES=${SEMANTIC_PROCESSES:-${#SEMANTIC_DEVICE_LIST[@]}}
if [[ "$DETECT_PROCESSES" -le 0 || "$SEMANTIC_PROCESSES" -le 0 ]]; then
    echo "Both detector and semantic worker need at least one GPU" >&2
    exit 2
fi

THRESHOLD=$(
    "$PYTHON_BIN" -c \
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

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS=(--overwrite)
fi

run_stream() {
    mkdir -p "$RUN_ROOT/queue/pending" "$RUN_ROOT/queue/processing" \
        "$RUN_ROOT/queue/done" "$RUN_ROOT/queue/producers"
    find "$RUN_ROOT/queue/producers" -maxdepth 1 -type f -name 'rank_*.done.json' -delete

    echo "[stream:qwen35] detector_gpus=$DETECT_GPUS semantic_gpus=$SEMANTIC_GPUS"
    echo "[stream:qwen35] threshold=$THRESHOLD run_root=$RUN_ROOT"

    CUDA_VISIBLE_DEVICES="$SEMANTIC_GPUS" \
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node "$SEMANTIC_PROCESSES" \
        --master_port "$SEMANTIC_MASTER_PORT" \
        "$VSR_ROOT/scripts/stream_caption_events_qwen35.py" \
        --model "$MODEL" \
        --run-root "$RUN_ROOT" \
        --producer-count "$DETECT_PROCESSES" \
        --dtype "${DTYPE:-bfloat16}" &
    SEMANTIC_PID=$!

    CUDA_VISIBLE_DEVICES="$DETECT_GPUS" \
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node "$DETECT_PROCESSES" \
        --master_port "$DETECT_MASTER_PORT" \
        "$VSR_ROOT/scripts/stream_detect_events_qwen35.py" \
        --model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --dataset-root "$DATASET_ROOT" \
        --source-parquet "$SOURCE_PARQUET" \
        --run-root "$RUN_ROOT" \
        --threshold "$THRESHOLD" \
        --top-fraction "${TOP_FRACTION:-0.05}" \
        --layer-fusion "${LAYER_FUSION:-mean}" \
        --dtype "${DTYPE:-bfloat16}" \
        "${RANGE_ARGS[@]}" "${OVERWRITE_ARGS[@]}" &
    DETECT_PID=$!

    cleanup_stream() {
        kill "$DETECT_PID" "$SEMANTIC_PID" 2>/dev/null || true
    }
    trap cleanup_stream INT TERM

    if ! wait "$DETECT_PID"; then
        echo "Qwen3.5 detector failed; stopping semantic workers" >&2
        kill "$SEMANTIC_PID" 2>/dev/null || true
        wait "$SEMANTIC_PID" 2>/dev/null || true
        exit 1
    fi
    if ! wait "$SEMANTIC_PID"; then
        echo "Qwen3.5 semantic workers failed" >&2
        exit 1
    fi
    trap - INT TERM

    "$PYTHON_BIN" "$VSR_ROOT/scripts/finalize_stream_event_memory.py" \
        --run-root "$RUN_ROOT"
}

run_qa() {
    QA_SELECTION_ARGS=(
        --event-retrieval "$QA_EVENT_RETRIEVAL"
        --memory-view "$QA_MEMORY_VIEW"
    )
    if [[ -n "${QA_EVENT_SCORE_THRESHOLD:-}" ]]; then
        QA_SELECTION_ARGS+=(--event-score-threshold "$QA_EVENT_SCORE_THRESHOLD")
    fi
    CUDA_VISIBLE_DEVICES="$SEMANTIC_GPUS" \
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node "$SEMANTIC_PROCESSES" \
        --master_port "$QA_MASTER_PORT" \
        "$VSR_ROOT/scripts/evaluate_event_memory_qwen35.py" \
        --model "$MODEL" \
        --memory-root "$QA_MEMORY_ROOT" \
        --source-parquet "$SOURCE_PARQUET" \
        --output-dir "$QA_OUTPUT_DIR" \
        --dtype "${DTYPE:-bfloat16}" \
        "${QA_SELECTION_ARGS[@]}" \
        "${RANGE_ARGS[@]}"
}

cd "$VSR_ROOT"
case "$MODE" in
    stream)
        run_stream
        ;;
    finalize)
        "$PYTHON_BIN" scripts/finalize_stream_event_memory.py --run-root "$RUN_ROOT"
        ;;
    qa)
        run_qa
        "$PYTHON_BIN" scripts/validate_event_memory_pipeline.py \
            --run-root "$RUN_ROOT" \
            --semantic-root "$QA_MEMORY_ROOT" \
            --qa-path "$QA_OUTPUT_DIR/results.json" \
            --output-path "$QA_OUTPUT_DIR/validation_report.json" \
            --require-qa
        ;;
    all)
        run_stream
        run_qa
        "$PYTHON_BIN" scripts/validate_event_memory_pipeline.py \
            --run-root "$RUN_ROOT" \
            --semantic-root "$QA_MEMORY_ROOT" \
            --qa-path "$QA_OUTPUT_DIR/results.json" \
            --output-path "$QA_OUTPUT_DIR/validation_report.json" \
            --require-qa
        ;;
esac
