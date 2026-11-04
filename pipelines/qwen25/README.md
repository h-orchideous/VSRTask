# Qwen2.5 Version

This directory is the isolated Qwen2.5 experiment entry point.

```text
Qwen2.5-VL frozen K/V (28 x 391 x 512)
  -> TemporalPatchDiffDetector
  -> Qwen2.5-VL semantic writer
  -> ordered explicit event memory
  -> Qwen2.5-VL VSR QA
```

The validated detector is
`/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt`.
It was trained on samples 1-300 for 10 epochs. Use `patchdiff.sh` for K/V diff,
cache, training, and detector evaluation. Use `official_vsr.sh all` for the
official streaming VSR pipeline.

```bash
./pipelines/qwen25/patchdiff.sh eval
./pipelines/qwen25/official_vsr.sh all
```

This version must not use a Qwen3.5 checkpoint, feature cache, threshold, writer,
or QA model.
