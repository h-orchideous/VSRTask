#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python}
MODEL=${MODEL:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:-$VSR_ROOT/outputs/vsi_super_recall_stream_memory_qwen25_10mins_v2}
SEMANTIC_GPUS=${SEMANTIC_GPUS:-0}
DTYPE=${DTYPE:-bfloat16}

export CUDA_VISIBLE_DEVICES="$SEMANTIC_GPUS"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VSR_ROOT:$VSR_ROOT/scripts:${PYTHONPATH:-}"

IFS=',' read -r -a DEVICES <<< "$SEMANTIC_GPUS"
SEMANTIC_PROCESSES=${SEMANTIC_PROCESSES:-${#DEVICES[@]}}
PRODUCER_COUNT=$(find "$RUN_ROOT/queue/producers" -maxdepth 1 \
    -type f -name 'rank_*.done.json' | wc -l)
if [[ "$PRODUCER_COUNT" -le 0 ]]; then
    echo "No producer completion markers under $RUN_ROOT/queue/producers" >&2
    exit 1
fi

cd "$VSR_ROOT"
echo "[repair] run_root=$RUN_ROOT semantic_gpus=$SEMANTIC_GPUS producers=$PRODUCER_COUNT"
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$SEMANTIC_PROCESSES" \
    scripts/stream_caption_events_qwen25.py \
    --model "$MODEL" \
    --run-root "$RUN_ROOT" \
    --producer-count "$PRODUCER_COUNT" \
    --dtype "$DTYPE"

"$PYTHON_BIN" scripts/finalize_stream_event_memory.py --run-root "$RUN_ROOT"
"$PYTHON_BIN" scripts/validate_event_memory_pipeline.py \
    --run-root "$RUN_ROOT" \
    --require-qa
