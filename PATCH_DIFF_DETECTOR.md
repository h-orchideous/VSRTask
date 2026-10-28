# Patch Diff Detector

This experiment is independent from the full Delta K/V regression model.

## Task

The frozen Qwen K/V cache provides two inputs:

```text
K, V: [B, 8, 28, 391, 512]
```

The detector concatenates K/V, applies a shared two-layer temporal
Transformer independently to every `(Qwen layer, patch)` stream, and predicts:

```text
patch_diff: [B, 28, 391]
```

For an edited current frame, the target is:

```text
sqrt(0.5 * (mean(delta_K^2, dim=-1) + mean(delta_V^2, dim=-1)))
```

For a clean current frame, the target is all zeros. Training uses MSE over all
Qwen layers and patches. At inference, layer scores are fused into `[B, 391]`
and the highest-scoring patches form the anomaly mask.

## Files

```text
vsrtask/patch_diff_detector.py          model and target definition
scripts/train_patch_diff_detector.py   cached-K/V distributed training
scripts/evaluate_patch_diff_detector.py held-out evaluation
scripts/visualize_patch_diff_masks.py  mask and oracle visualization
```

The original checkpoint format is supported without conversion.

## Four-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_patch_diff_detector.py \
  --kv-diff-root /data1/ZhangHuayu/datasets/VSR-Train/kv_diff \
  --kv-cache-root /data3/ZhangHuayu/VSR-Train/kv_features \
  --frames-root /data1/ZhangHuayu/tmp/vsr_train_frames \
  --output-dir /data1/ZhangHuayu/checkpoints/vsr_patch_diff_clean \
  --sample-range 1 300 \
  --window-size 8 \
  --negative-windows 4 \
  --batch-size 8 \
  --detector-d-model 256 \
  --detector-layers 2 \
  --detector-heads 8 \
  --dtype bfloat16 \
  --epochs 10 \
  --seed 42 \
  --progress-every 1 \
  --log-every 100
```

## Held-Out Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/python \
  scripts/evaluate_patch_diff_detector.py \
  --checkpoint /data1/ZhangHuayu/checkpoints/vsr_patch_diff_clean/checkpoint-10.pt \
  --kv-diff-root /data1/ZhangHuayu/datasets/VSR-Train/kv_diff \
  --kv-cache-root /data3/ZhangHuayu/VSR-Train/kv_features_val \
  --frames-root /data1/ZhangHuayu/tmp/vsr_train_frames \
  --sample-range 301 310 \
  --negative-windows 4 \
  --top-fraction 0.1 \
  --output-json /data1/ZhangHuayu/checkpoints/vsr_patch_diff_clean/eval_301_310.json
```
