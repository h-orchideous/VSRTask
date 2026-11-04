#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-help}
VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}
export MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
export KV_DIFF_ROOT=${KV_DIFF_ROOT:-/data1/ZhangHuayu/datasets/VSR-Train/kv_diff_qwen35}
export KV_CACHE_ROOT=${KV_CACHE_ROOT:-/data3/ZhangHuayu/VSR-Train/kv_features/qwen35_full_attention}
export OUTPUT_DIR=${OUTPUT_DIR:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen35_1_377_epoch10_4gpu}
export SAMPLE_START=${SAMPLE_START:-1}
export SAMPLE_END=${SAMPLE_END:-377}

case "$MODE" in
    diff|cache|train)
        exec "$VSR_ROOT/scripts/run_qwen35_patch_diff_pipeline.sh" "$MODE"
        ;;
    eval)
        CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} "$PYTHON_BIN" \
            "$VSR_ROOT/scripts/evaluate_patch_diff_detector.py" \
            --checkpoint "${CHECKPOINT:-$OUTPUT_DIR/checkpoint-10.pt}" \
            --kv-diff-root "$KV_DIFF_ROOT" \
            --kv-cache-root "$KV_CACHE_ROOT" \
            --frames-root "${FRAMES_ROOT:-/data1/ZhangHuayu/tmp/vsr_train_frames}" \
            --sample-range "$SAMPLE_START" "$SAMPLE_END" \
            --negative-windows 4 \
            --top-fraction "${TOP_FRACTION:-0.05}" \
            --output-json "${EVAL_OUTPUT:-$VSR_ROOT/outputs/qwen35_patchdiff_eval.json}"
        ;;
    calibrate)
        "$PYTHON_BIN" "$VSR_ROOT/scripts/calibrate_window_event_threshold.py" \
            --evaluation-results "${EVAL_OUTPUT:-$VSR_ROOT/outputs/qwen35_patchdiff_eval.json}" \
            --target-recall "${TARGET_RECALL:-1.0}" \
            --output "${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_qwen35.json}"
        ;;
    *)
        echo "Usage: $0 {diff|cache|train|eval|calibrate}" >&2
        exit 2
        ;;
esac
