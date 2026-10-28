#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
CAMBRIAN_ROOT=/home/ZhangHuayu/Workspace/cambrian-s
DATA_ROOT=${DATA_ROOT:-$VSR_ROOT/outputs/vsr_gt_top4_301_350}
OUTPUT_PATH=${OUTPUT_PATH:-$VSR_ROOT/outputs/vsr_gt_top4_301_350_qwen_eval}
TASK_PATH=$VSR_ROOT/lmms_tasks
TASK=vsr_gt_top4_301_350
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VSI_SUPER_RECALL_ROOT="$DATA_ROOT"
export VSR_SAMPLE_START=301
export VSR_SAMPLE_END=350
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH="$LMMS_ROOT:$CAMBRIAN_ROOT:$VSR_ROOT:${PYTHONPATH:-}"

PARQUET=$DATA_ROOT/test_10mins_301_350.parquet
test -f "$PARQUET" || {
    echo "Missing GT Top-4 dataset: $PARQUET" >&2
    exit 2
}
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}
MODEL_ARGS=${MODEL_ARGS:-"pretrained=$CHECKPOINT,min_pixels=100352,max_pixels=100352,max_num_frames=4,fps=1,use_custom_video_loader=True,max_image_size=448,attn_implementation=sdpa,interleave_visuals=False,use_fast_processor=True"}

mkdir -p "$OUTPUT_PATH"
echo "[qwen-gt-top4] samples=301-350 frames_per_video=4"
echo "[qwen-gt-top4] data=$DATA_ROOT output=$OUTPUT_PATH"

cd "$LMMS_ROOT"
"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "${MAIN_PROCESS_PORT:-29763}" \
    -m lmms_eval \
    --model qwen2_5_vl \
    --force_simple \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --include_path "$TASK_PATH" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gt_top4_301_350_qwen \
    --output_path "$OUTPUT_PATH"
