#!/usr/bin/env bash
set -euo pipefail

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
CAMBRIAN_ROOT=/home/ZhangHuayu/Workspace/cambrian-s
DATA_ROOT=${DATA_ROOT:-$VSR_ROOT/outputs/vsr_gt_top4_301_350}
OUTPUT_PATH=${OUTPUT_PATH:-$VSR_ROOT/outputs/vsr_gt_top4_images_301_350_qwen_eval}
TASK_PATH=$VSR_ROOT/lmms_tasks
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VSR_GT_TOP4_BUILD_REPORT="$DATA_ROOT/build_report.json"
export VSR_SAMPLE_START=${VSR_SAMPLE_START:-301}
export VSR_SAMPLE_END=${VSR_SAMPLE_END:-350}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH="$LMMS_ROOT:$CAMBRIAN_ROOT:$VSR_ROOT:${PYTHONPATH:-}"

test -f "$VSR_GT_TOP4_BUILD_REPORT" || {
    echo "Missing build report: $VSR_GT_TOP4_BUILD_REPORT" >&2
    exit 2
}
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}
MODEL_ARGS=${MODEL_ARGS:-"pretrained=$CHECKPOINT,min_pixels=100352,max_pixels=100352,max_num_frames=4,fps=1,use_custom_video_loader=True,max_image_size=448,attn_implementation=sdpa,interleave_visuals=False,use_fast_processor=True"}

mkdir -p "$OUTPUT_PATH"
echo "[qwen-gt-top4-images] samples=$VSR_SAMPLE_START-$VSR_SAMPLE_END images_per_sample=4"
echo "[qwen-gt-top4-images] bypassing MP4 decode and video-frame sampling"

cd "$LMMS_ROOT"
"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "${MAIN_PROCESS_PORT:-29773}" \
    -m lmms_eval \
    --model qwen2_5_vl \
    --force_simple \
    --model_args "$MODEL_ARGS" \
    --tasks vsr_gt_top4_images_301_350 \
    --include_path "$TASK_PATH" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gt_top4_images_301_350_qwen \
    --output_path "$OUTPUT_PATH"
