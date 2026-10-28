#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-both}

VSR_ROOT=/home/ZhangHuayu/Workspace/VSRTask
LMMS_ROOT=/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
CAMBRIAN_ROOT=/home/ZhangHuayu/Workspace/cambrian-s
PYTHON_BIN=${PYTHON_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-/home/ZhangHuayu/miniconda3/envs/cambrians_eval/bin/accelerate}
CHECKPOINT=${CHECKPOINT:-/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct}
TASK_PATH=$VSR_ROOT/lmms_tasks
TASK=vsr_mask_ab_301_350

ORIGINAL_ROOT=${ORIGINAL_ROOT:-/data1/ZhangHuayu/datasets/VSR-Train}
MASKED_ROOT=${MASKED_ROOT:-$VSR_ROOT/outputs/vsr_checkpoint10_top5_masked_301_350}
OUTPUT_BASE=${OUTPUT_BASE:-$VSR_ROOT/outputs/ssm_mask_ab_checkpoint10_top5}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-20480}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH="$LMMS_ROOT:$CAMBRIAN_ROOT:$VSR_ROOT:${PYTHONPATH:-}"
export VSR_SAMPLE_START=${VSR_SAMPLE_START:-301}
export VSR_SAMPLE_END=${VSR_SAMPLE_END:-350}

IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_PROCESSES=${NUM_PROCESSES:-${#DEVICES[@]}}

MODEL_ARGS="pretrained=$CHECKPOINT,min_pixels=100352,max_pixels=100352,max_num_frames=-1,fps=1,use_custom_video_loader=True,max_image_size=448,attn_implementation=sdpa,interleave_visuals=False,use_fast_processor=True,ssm_d_state=64,ssm_max_memory_len=256,ssm_fusion_num_heads=8,ssm_fusion_bottleneck=256,ssm_layer_sharing=group4,ssm_fusion_policy=query_only,ssm_frame_window=128,ssm_prefix_len=0,ssm_enabled=true"

run_one() {
    local name=$1
    local video_root=$2
    local port=$3
    local output_dir="$OUTPUT_BASE/$name"

    mkdir -p "$output_dir"
    export VSI_SUPER_RECALL_ROOT="$video_root"

    echo "[ssm-mask-ab] run=$name"
    echo "[ssm-mask-ab] video_root=$VSI_SUPER_RECALL_ROOT"
    echo "[ssm-mask-ab] output=$output_dir"
    echo "[ssm-mask-ab] samples=$VSR_SAMPLE_START-$VSR_SAMPLE_END"
    echo "[ssm-mask-ab] gpus=$CUDA_VISIBLE_DEVICES"

    cd "$LMMS_ROOT"
    "$ACCELERATE_BIN" launch \
        --num_processes "$NUM_PROCESSES" \
        --main_process_port "$port" \
        -m lmms_eval \
        --model qwen2_5_ssm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --include_path "$TASK_PATH" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_ssm_${name}" \
        --output_path "$output_dir"
}

case "$MODE" in
    original)
        run_one original "${ORIGINAL_ROOT}" "${MAIN_PROCESS_PORT:-29631}"
        ;;
    masked)
        run_one masked_top5 "${MASKED_ROOT}" "${MAIN_PROCESS_PORT:-29632}"
        ;;
    both)
        run_one original "${ORIGINAL_ROOT}" "${MAIN_PROCESS_PORT_ORIGINAL:-29631}"
        run_one masked_top5 "${MASKED_ROOT}" "${MAIN_PROCESS_PORT_MASKED:-29632}"
        ;;
    *)
        echo "Usage: $0 [original|masked|both]" >&2
        exit 2
        ;;
esac
