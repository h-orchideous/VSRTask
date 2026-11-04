#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/qwen35_vlm/bin/python}
export MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen3.5-9B}
export CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen35_1_377_epoch10_4gpu/checkpoint-10.pt}
export THRESHOLD_CONFIG=${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_qwen35.json}
export RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen35_10mins}

exec "$VSR_ROOT/scripts/run_stream_event_memory_qwen35.sh" "${1:-all}"
