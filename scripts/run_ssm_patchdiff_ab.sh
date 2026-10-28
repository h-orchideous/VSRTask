#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-both}
DURATION=${DURATION:-10mins}

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSI-SUPER-Recall}
OUTPUT_BASE=${OUTPUT_BASE:-$VSR_ROOT/outputs/ssm_patchdiff_ab}
TASK=cambrians_vsr_local_${DURATION}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}
export VSI_SUPER_RECALL_ROOT=$DATASET_ROOT

run_baseline() {
    TASK="$TASK" \
    MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT_BASELINE:-29751} \
    OUTPUT_PATH="$OUTPUT_BASE/$DURATION/ssm" \
    "$LMMS_ROOT/scripts/vsr_qwen25ssm.sh"
}

run_patchdiff() {
    TASK="$TASK" \
    MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT_PATCHDIFF:-29752} \
    PATCH_DETECTOR_CHECKPOINT=/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt \
    PATCH_DETECTOR_THRESHOLD=${PATCH_DETECTOR_THRESHOLD:-0.45} \
    PATCH_DETECTOR_GAIN=${PATCH_DETECTOR_GAIN:-1.0} \
    PATCH_DETECTOR_MODE=${PATCH_DETECTOR_MODE:-continuous} \
    OUTPUT_PATH="$OUTPUT_BASE/$DURATION/ssm_patchdiff" \
    "$LMMS_ROOT/scripts/vsr_qwen25ssm_patch_detector.sh"
}

case "$MODE" in
    baseline) run_baseline ;;
    patchdiff) run_patchdiff ;;
    both)
        run_baseline
        run_patchdiff
        ;;
    *)
        echo "Usage: $0 [baseline|patchdiff|both]" >&2
        exit 2
        ;;
esac
