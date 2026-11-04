# Qwen3.5 Version

This directory is the isolated Qwen3.5 experiment entry point.

```text
Qwen3.5-9B frozen full-attention K/V (8 x 300 x 1024)
  -> TemporalPatchDiffDetector
  -> Qwen3.5-9B semantic writer
  -> ordered explicit event memory
  -> Qwen3.5-9B VSR QA
```

The detector checkpoint is
`/data1/ZhangHuayu/checkpoints/vsr_patch_diff_qwen35_1_377_epoch10_4gpu/checkpoint-10.pt`.
It was trained on samples 1-377 for 10 epochs. The 24 linear-attention layers
are intentionally excluded because their recurrent states are not standard K/V
tensors. Use `patchdiff.sh` for K/V diff, cache, and training. Use
`official_vsr.sh all` after generating `configs/event_threshold_qwen35.json`.

```bash
SAMPLE_START=301 SAMPLE_END=350 ./pipelines/qwen35/patchdiff.sh eval
./pipelines/qwen35/patchdiff.sh calibrate
./pipelines/qwen35/official_vsr.sh all
```

The default Qwen3.5 threshold calibration targets 100% recall on cached edited
windows. False positives are acceptable at this stage because missing an event
is irreversible, while extra memory events can be filtered during retrieval.

This version must not use a Qwen2.5 checkpoint, feature cache, threshold, writer,
or QA model.

## Current Verification

Checkpoint-10 cached-window evaluation on samples 301-350:

| Metric | Value |
|---|---:|
| Overall MSE | 0.1961 |
| Positive MSE | 0.2994 |
| Negative MSE | 0.0927 |
| Positive predicted mean | 0.7024 |
| Negative predicted mean | 0.1603 |
| Top-5% patch IoU | 0.1474 |

The operational threshold is `1.0890907049179077`. It reaches 108/108 edited
windows on the 351-377 calibration subset with 0/108 sampled clean windows
triggered. This is a high-recall training-distribution calibration, not a
held-out full-frame estimate.

On official sample `00000000`, threshold `1.24255` wrote three events and
answered incorrectly. Threshold `1.08909` wrote the four ordered Pikachu events
at frames 85, 89, 143, and 198 and answered correctly. The full 60-video run is
stored under `outputs/vsi_super_recall_stream_memory_qwen35_10mins_t1089_full`.
