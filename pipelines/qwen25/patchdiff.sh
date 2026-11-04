#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-help}
VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSR-Train}
FRAMES_ROOT=${FRAMES_ROOT:-/data1/ZhangHuayu/tmp/vsr_train_frames}
EDITED_FRAMES_ROOT=${EDITED_FRAMES_ROOT:-/data1/ZhangHuayu/tmp/vsr_edited_frames}
KV_DIFF_ROOT=${KV_DIFF_ROOT:-$DATASET_ROOT/kv_diff}
KV_CACHE_ROOT=${KV_CACHE_ROOT:-/data3/ZhangHuayu/VSR-Train/kv_features}
OUTPUT_DIR=${OUTPUT_DIR:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen25_epoch10}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt}
SAMPLE_START=${SAMPLE_START:-1}
SAMPLE_END=${SAMPLE_END:-300}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-4}

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"
cd "$VSR_ROOT"

case "$MODE" in
    diff)
        CUDA_VISIBLE_DEVICES=${DIFF_GPU:-0} "$PYTHON_BIN" \
            /data1/ZhangHuayu/tools/compute_vsr_kv_diff.py \
            --manifest "$DATASET_ROOT/manifest.json" \
            --frames-root "$FRAMES_ROOT" \
            --edited-frames-dir "$EDITED_FRAMES_ROOT" \
            --output-root "$KV_DIFF_ROOT" \
            --model "$MODEL" \
            --sample-range "$SAMPLE_START" "$SAMPLE_END" \
            --dtype bfloat16
        ;;
    cache)
        "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
            scripts/cache_vsr_kv_features.py \
            --model "$MODEL" \
            --backbone-name Qwen2.5-VL-7B-Instruct \
            --kv-diff-root "$KV_DIFF_ROOT" \
            --frames-root "$FRAMES_ROOT" \
            --manifest "$DATASET_ROOT/manifest.json" \
            --dataset-root "$DATASET_ROOT" \
            --output-root "$KV_CACHE_ROOT" \
            --sample-range "$SAMPLE_START" "$SAMPLE_END" \
            --window-size 8 \
            --negative-windows 4 \
            --cache-epochs "${CACHE_EPOCHS:-10}" \
            --dtype float16
        ;;
    train)
        "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
            scripts/train_patch_diff_detector.py \
            --kv-diff-root "$KV_DIFF_ROOT" \
            --kv-cache-root "$KV_CACHE_ROOT" \
            --frames-root "$FRAMES_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            --sample-range "$SAMPLE_START" "$SAMPLE_END" \
            --window-size 8 \
            --negative-windows 4 \
            --batch-size "${BATCH_SIZE:-8}" \
            --detector-d-model 256 \
            --detector-layers 2 \
            --detector-heads 8 \
            --dtype bfloat16 \
            --epochs "${EPOCHS:-10}" \
            --seed 42 \
            --progress-every 1 \
            --log-every 100
        ;;
    eval)
        CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} "$PYTHON_BIN" \
            scripts/evaluate_patch_diff_detector.py \
            --checkpoint "$CHECKPOINT" \
            --kv-diff-root "$KV_DIFF_ROOT" \
            --kv-cache-root "$KV_CACHE_ROOT" \
            --frames-root "$FRAMES_ROOT" \
            --sample-range "$SAMPLE_START" "$SAMPLE_END" \
            --negative-windows 4 \
            --top-fraction "${TOP_FRACTION:-0.05}" \
            --output-json "${EVAL_OUTPUT:-$VSR_ROOT/outputs/qwen25_patchdiff_eval.json}"
        ;;
    calibrate)
        "$PYTHON_BIN" scripts/calibrate_window_event_threshold.py \
            --evaluation-results "${EVAL_OUTPUT:-$VSR_ROOT/outputs/qwen25_patchdiff_eval.json}" \
            --target-recall "${TARGET_RECALL:-0.90}" \
            --output "${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_qwen25_windows.json}"
        ;;
    *)
        echo "Usage: $0 {diff|cache|train|eval|calibrate}" >&2
        exit 2
        ;;
esac
