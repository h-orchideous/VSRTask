#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
DATASET_ROOT=${DATASET_ROOT:-/data1/ZhangHuayu/datasets/VSR-Train}
FRAMES_ROOT=${FRAMES_ROOT:-/data1/ZhangHuayu/tmp/vsr_train_frames}
EDITED_FRAMES_ROOT=${EDITED_FRAMES_ROOT:-/data1/ZhangHuayu/tmp/vsr_edited_frames}
KV_DIFF_ROOT=${KV_DIFF_ROOT:-$DATASET_ROOT/kv_diff_qwen35}
KV_CACHE_ROOT=${KV_CACHE_ROOT:-/data3/ZhangHuayu/VSR-Train/kv_features/qwen35_full_attention}
OUTPUT_DIR=${OUTPUT_DIR:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen35_epoch10}
SAMPLE_START=${SAMPLE_START:-1}
SAMPLE_END=${SAMPLE_END:-300}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-4}

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:${PYTHONPATH:-}"
cd "$VSR_ROOT"

MODE=${1:-help}
case "$MODE" in
  diff)
    # Diff extraction currently runs on one GPU; each sample has only four
    # edited events. Split sample ranges manually for concurrent jobs if needed.
    CUDA_VISIBLE_DEVICES=${DIFF_GPU:-0} "$PYTHON_BIN" \
      scripts/compute_vsr_kv_diff_qwen35.py \
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
      scripts/cache_vsr_kv_features_qwen35.py \
      --model "$MODEL" \
      --backbone-name Qwen3.5-9B-full-attention \
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
      --epochs 10 \
      --seed 42 \
      --progress-every 1 \
      --log-every 100
    ;;
  *)
    echo "Usage: $0 {diff|cache|train}" >&2
    exit 2
    ;;
esac
