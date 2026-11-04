#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
export PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
export MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
export CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt}
export THRESHOLD_CONFIG=${THRESHOLD_CONFIG:-$VSR_ROOT/configs/event_threshold_balanced.json}
export RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen25_10mins}

exec "$VSR_ROOT/scripts/run_stream_event_memory_qwen25.sh" "${1:-all}"
