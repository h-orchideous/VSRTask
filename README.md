# VSRTask

Patch-difference detection and explicit event memory for long-video spatial
and temporal reasoning.

## Versioned Pipelines

The model-specific systems are isolated under two entry directories:

| Version | Detection features | Semantic writer | QA | Entry directory |
|---|---|---|---|---|
| Qwen2.5 | Qwen2.5-VL 28-layer K/V | Qwen2.5-VL | Qwen2.5-VL | `pipelines/qwen25` |
| Qwen3.5 | Qwen3.5 8 full-attention-layer K/V | Qwen3.5 | Qwen3.5 | `pipelines/qwen35` |

Each directory fixes its own model, environment, checkpoint, feature cache,
threshold, training entry point, and official VSR entry point. Shared detector,
data, and memory-schema implementations remain under `vsrtask/` and `scripts/`.
Do not mix checkpoints, K/V caches, or thresholds across the two versions.

## Official 10-Minute Result Comparison

The table below records completed end-to-end runs on the 60 samples in the
official VSI-SUPER-Recall `test_10mins.parquet`. Detection, semantic writing,
and QA use the same Qwen version within each row. These are full streaming
pipelines: every video frame is scanned before the query is exposed, events
crossing the fixed threshold are written immediately, and QA reads the ordered
text memory only.

| Pipeline | Detector threshold | Written events | Zero-event videos | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 PatchDiff + Qwen2.5 writer/QA (`v2`) | `0.9073798` | 236 | 1 / 60 | **36 / 60** | **60.00%** |
| Qwen3.5 PatchDiff + Qwen3.5 writer/QA (`t1089`) | `1.0890907` | 197 | 7 / 60 | 29 / 60 | 48.33% |

The completed Qwen3.5 run is stored under:

```text
outputs/vsi_super_recall_stream_memory_qwen35_10mins_t1089_full
```

Its queue and memory are complete: 197 detector events produced 197 validated
semantic events, all 60 per-video memory files were assembled, the QA memory
digest matches the current memory, and `validation_report.json` reports
`valid=true`. One event initially failed because the contamination filter
mistook the substring `adde` in `ladder` for a control token. The filter is now
case-sensitive for camel-case control names, the event was regenerated, and a
regression test covers `on the ladder`.

The Qwen3.5 result is strongly associated with event recall:

| Events written for one video | Videos | Correct | Accuracy |
|---|---:|---:|---:|
| 0 | 7 | 1 | 14.29% |
| 1-3 | 19 | 5 | 26.32% |
| Exactly 4 | 24 | 17 | 70.83% |
| More than 4 | 10 | 6 | 60.00% |

Thus 36/60 videos do not receive exactly four events, and these videos account
for most errors. A lower Qwen3.5 threshold of `1.0` is being evaluated in a
separate output directory so the `t1089` result remains reproducible:

```text
config:  configs/event_threshold_qwen35_t1000.json
output:  outputs/vsi_super_recall_stream_memory_qwen35_10mins_t1000_full
log:     logs/qwen35_official_10mins_t1000_full.log
```

Before semantic generation, replaying `1.0` over the saved frame-score streams
predicts 362 candidates, 47/60 videos with at least four candidates, and two
zero-event videos. These are candidate-count diagnostics, not the final `1.0`
QA result; the row will be added to the completed table only after the new
memory and QA validation finish.

## Validated Qwen2.5 + Qwen2.5 System

The currently validated result uses **Qwen2.5-VL for both sides of the
pipeline**. It is not a Qwen3.5 result:

```text
video frame
  -> frozen Qwen2.5-VL-7B-Instruct K/V features
  -> trained PatchDiff detector
  -> threshold-triggered anomaly event
  -> Qwen2.5-VL-7B-Instruct semantic writer
  -> ordered explicit event memory
  -> Qwen2.5-VL-7B-Instruct multiple-choice QA
```

More precisely:

- **Detection feature backbone:**
  `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct`.
- **Anomaly detector:** the 10-epoch PatchDiff checkpoint
  `/data1/ZhangHuayu/checkpoints/vsr_patch_diff_resumed_epoch10/checkpoint-10.pt`.
- **Semantic memory writer:**
  `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct`.
- **Final question-answering model:**
  `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct`.
- Qwen2.5-VL is frozen during PatchDiff training and event detection. The
  trainable component is the small temporal PatchDiff detector.
- The Qwen3.5 K/V-diff branch under development is a separate experiment and
  is not used in the result below.

### Streaming Memory Write Procedure

The validated run is query-independent until final QA. The VSR question is
not exposed to the detector or semantic writer.

1. Decode the official 10-minute video sequentially at its stored 1 FPS.
2. For every current frame, extract Qwen2.5 K/V features and run the PatchDiff
   detector on an 8-frame causal window: the current frame plus seven previous
   frames.
3. Fuse the predicted per-layer patch differences by taking the mean across
   the 28 Qwen layers. The frame anomaly score is the mean of the highest 5%
   of the 391 visual-token scores, approximately 20 tokens.
4. If the frame score is at least `0.9073798060417175`, immediately create one
   independent event. This run does not use per-video Top-4 selection,
   temporal merging, or an event-count cap. Adjacent threshold crossings are
   retained as separate events so their order cannot be collapsed.
5. Persist the detector evidence: frame/time index, anomaly score and margin,
   original frame, red Top-5% token overlay, selected token indices, token
   grid, and compressed scores for every scanned frame.
6. Atomically enqueue the event through `pending -> processing -> done`.
   Atomic rename lets semantic workers claim each event once and makes the
   queue restartable.
7. A separate Qwen2.5-VL semantic worker reads the **original selected frame
   and an unmodified crop around the highest-scoring PatchDiff token**, without
   seeing the downstream VSR question. The red overlay remains audit evidence
   and is never sent to the semantic model. The clean crop directs attention
   without introducing artificial objects. Schema v3 records the anomaly,
   grounded physical placement, and full-frame context separately. It does not
   infer the question, hidden cause, or temporal change.
8. Because multiple semantic workers can finish out of order, completed
   events are assembled by `peak_frame` into one chronological memory file
   per video. No event is removed during assembly.
9. Only after memory construction is complete is the VSR question introduced.
   Qwen2.5-VL answers from the ordered textual event memory.

A semantic event has the following effective structure:

```json
{
  "event_id": 1,
  "temporal_order": 1,
  "peak_frame": 85,
  "peak_seconds": 85.0,
  "peak_score": 1.3574,
  "score_margin": 0.4500,
  "episodic_memory": {
    "when": {"frame": 85, "seconds": 85.0},
    "semantic_schema_version": 3,
    "what": "A stationary Pikachu plush toy",
    "where": "inside a trash can",
    "scene": "Living room",
    "background": "a trash can beside a sofa and table",
    "objects": [
      {"name": "pikachu plush toy", "count": 1},
      {"name": "trash can", "count": 1},
      {"name": "sofa", "count": 1}
    ],
    "anomaly": {
      "entity": "pikachu plush toy",
      "description": "A stationary Pikachu plush toy",
      "count": 1
    },
    "placement": {
      "relation": "inside",
      "anchor": "a trash can",
      "text": "inside a trash can",
      "status": "direct",
      "observation_text": "inside a trash can"
    },
    "context": {
      "scene": "Living room",
      "summary": "a trash can beside a sofa and table",
      "objects": [
        {"name": "trash can", "count": 1},
        {"name": "sofa", "count": 1}
      ]
    },
    "semantic_writer": {
      "model": "/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct",
      "query_conditioned": false,
      "streaming": true
    }
  },
  "evidence": {
    "original_image": ".../frame_000085_original.jpg",
    "overlay_image": ".../frame_000085_overlay.jpg",
    "selected_patch_indices": [128, 83, 186],
    "token_grid": [17, 23]
  }
}
```

### Official 10-Minute Validation Result

Run directory:

```text
outputs/vsi_super_recall_stream_memory_qwen25_10mins_v2
```

Dataset and execution settings:

| Setting | Value |
|---|---:|
| Dataset | Official VSI-SUPER-Recall `test_10mins.parquet` |
| Videos / QA samples | 60 |
| Detector GPUs | 3 |
| Semantic writer / QA GPUs | 1 |
| Approximate wall time | 3.5 hours |
| Detection threshold | 0.9073798060417175 |
| Patch fraction used for score/evidence | 5% |

Memory-write statistics:

| Metric | Value |
|---|---:|
| Detection events | 236 |
| Semantic events written | 236 |
| Queue-lost events | 0 |
| Mean events per video | 3.93 |
| Videos with 0 events | 1 |
| Minimum / maximum events per video | 0 / 8 |
| Events with incomplete semantic fields | 14 / 236 (5.9%) |
| Videos containing incomplete events | 11 / 60 |

Of the 14 incomplete semantic events, 12 were missing
`anomaly_entity/anomaly_count` and two were missing `what`. Therefore
`validation_report.json` has `valid=false`; this indicates semantic schema
incompleteness, not queue loss. Detection and semantic event totals are both
236.

### Threshold For Subsequent Runs

The `0.9073798060417175` threshold above belongs to the recorded 60% baseline
and is not changed retroactively. New runs now default to
`configs/event_threshold_balanced.json`, whose threshold is
`0.8880652189254761`.

On the held-out 301-350 edited-frame scores, the new point changes recall from
`180/200 = 90.0%` to `181/200 = 90.5%`, with hard-negative precision `91.4%`.
Applied to the already stored official 10-minute score streams without using
their QA labels, it would trigger 286 events over 60 videos: mean 4.77, median
4, range 2-10, and no zero-event video. By comparison, lowering the threshold
to the 95%-recall point (`0.6641`) would trigger 2,755 events, or 45.9 per video,
which is too noisy and expensive for semantic memory writing.

The calibration utilities now support saving every frame score with
`evaluate_temporal_frame_detection.py --save-frame-scores`. Once that complete
clean-frame distribution has been generated, `calibrate_event_threshold.py`
can replace the balanced configuration using explicit recall, clean-frame FPR,
and precision constraints.

The original `v2` numbers above are preserved as the measured **legacy-writer
baseline**. The structured writer is a new semantic schema and must be evaluated
in a separate output directory before comparing accuracy. New events cannot
enter `done` unless the observation, structured placement, and visible-object
inventory are present and free of prompt/template contamination.
Malformed generations are retried; if the inventory omits the anomaly, it is
derived from the anomaly description with count one, which matches the current
single-insert construction. Existing legacy records can still be repaired
locally; incomplete records are returned from `done` to `pending` for model
regeneration.

Repair only incomplete events from an existing streaming run without rescanning
the videos:

```bash
RUN_ROOT=/home/ZhangHuayu/Workspace/VSRTask/outputs/vsi_super_recall_stream_memory_qwen25_10mins_v2 \
SEMANTIC_GPUS=0 \
./scripts/repair_stream_semantic_qwen25.sh
```

The repair command reprocesses only invalid event JSON files, reassembles the
chronological per-video memories, and runs semantic-quality validation. It does
not rerun QA, so `qa/results.json` continues to describe the previous memory
until QA is explicitly executed again. QA reruns constrain decoding to one of
`A/B/C/D` and archive the previous result files under `qa/archive/<timestamp>/`
before writing new results.

### Full-Frame Writer A/B Run

Do not overwrite the legacy semantic memories when evaluating the new writer.
Reuse the already-computed detector evidence, write a second semantic memory
set, then point QA at that set:

```bash
cd /home/ZhangHuayu/Workspace/VSRTask
conda activate vsr_train

RUN_ROOT=/home/ZhangHuayu/Workspace/VSRTask/outputs/vsi_super_recall_stream_memory_qwen25_10mins_v2
NEW_MEMORY_ROOT="$RUN_ROOT/semantic/memories_full_frame_v2"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/torchrun \
  --standalone --nproc_per_node=4 \
  scripts/caption_event_memory_qwen25.py \
  --model /data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct \
  --memory-root "$RUN_ROOT/detection/memories" \
  --output-root "$NEW_MEMORY_ROOT" \
  --dtype bfloat16 \
  --overwrite

DURATION=10mins \
RUN_ROOT="$RUN_ROOT" \
SEMANTIC_GPUS=0,1,2,3 \
QA_MEMORY_ROOT="$NEW_MEMORY_ROOT" \
QA_OUTPUT_DIR="$RUN_ROOT/qa_full_frame_v2" \
./scripts/run_stream_event_memory_qwen25.sh qa
```

The second command writes `qa_full_frame_v2/results.json` and
`qa_full_frame_v2/validation_report.json`. Compare those with the legacy
`qa/results.json`; detector events are identical, so this isolates the writing
and reading policy.

Final accuracy and baselines:

| Method | Correct | Accuracy |
|---|---:|---:|
| Original SSM on the complete video | 15 / 60 | 25.00% |
| SSM with Top-5% visual mask | 14 / 60 | 23.33% |
| PatchDiff Top-4 original frames -> Qwen2.5-VL | 18 / 60 | 30.00% |
| **Streaming PatchDiff + explicit memory -> Qwen2.5-VL** | **36 / 60** | **60.00%** |

Five QA generations did not produce a valid A/B/C/D option and were counted
as incorrect. Consequently, the 60% result includes both detector/writer
errors and unconstrained QA-format errors; it is not a filtered score.

The recorded artifacts are:

```text
detection/memory_manifest.json
detection/memories/<duration>/<video>.json
detection/evidence/<duration>/<video>/frame_*_{original,overlay}.jpg
detection/frame_scores/<duration>/<video>.jsonl.gz
semantic/semantic_memory_manifest.json
semantic/memories/<duration>/<video>.json
qa/results.json
validation_report.json
```

See `docs/event_memory_pipeline.md` for launcher and queue details.

### Memory QA Ablations (2026-10-30)

The following ablations reuse the same 60 official 10-minute videos, the same
Qwen2.5-VL QA model, and the same source semantic memory. They change only the
event selection, location text, or memory view supplied to QA.

| Memory supplied to QA | Events | Correct | Accuracy |
|---|---:|---:|---:|
| Balanced threshold (`0.8880652`), all events | 286 | 32 / 60 | 53.33% |
| Legacy threshold (`0.9073798`), all events | 236 | **35 / 60** | **58.33%** |
| Balanced threshold, strict target-entity retrieval | 172 | 32 / 60 | 53.33% |
| Balanced threshold, isolated physical-`where` rewrite | 286 | 30 / 60 | 50.00% |
| Balanced threshold, anomaly + placement only | 286 | 29 / 60 | 48.33% |
| Legacy threshold, anomaly + placement only | 236 | 30 / 60 | 50.00% |
| Legacy threshold, physical `where` + compact view | 236 | 31 / 60 | 51.67% |
| Legacy threshold, hierarchical context fallback | 236 | 30 / 60 | 50.00% |

The legacy-threshold filter improves four previously wrong samples and regresses
one, for a net gain of three. This confirms that the 50 additional low-score
events introduce useful recall but also more QA noise. Strict entity retrieval
does not improve aggregate accuracy: semantic aliases recover a target match in
all 60 videos, but 34 videos retain fewer than four events, so hard filtering
also removes useful but imperfectly captioned evidence.

The first physical-`where` experiment rewrote all semantic fields and scored
24/60; it is retained only as a rejected, confounded ablation. The corrected
experiment preserved `what`, `scene`, `background`, and `objects` byte-for-byte
and rewrote only `where`: 283/286 locations changed, no image-coordinate answer
survived validation, and accuracy was 30/60. Therefore physical-`where` remains
an optional experiment and is not enabled in the default writer.

Removing context is also rejected. Under the legacy threshold, compacting each
event to only anomaly and placement regresses from 35/60 to 30/60. Making scene
and background an explicit fallback also scores 30/60. The full view is helping
when the old `where` value is vague or wrong, so field formatting alone cannot
repair the memory. At the legacy threshold, 21/60 videos contain fewer than four
events; those samples score 8/21, compared with 27/39 for videos containing at
least four events. Missing events and incorrect semantic grounding are therefore
the main remaining errors.

### Structured Semantic Memory v3

Newly generated memories use a nested contract rather than adding more flat
fields:

```json
{
  "anomaly": {
    "entity": "pikachu",
    "description": "yellow Pikachu toy",
    "count": 1
  },
  "placement": {
    "relation": "on",
    "anchor": "the chair",
    "text": "on the chair",
    "status": "direct",
    "observation_text": "on the chair"
  },
  "context": {
    "scene": "bedroom",
    "summary": "a bed, nightstand, and closed door",
    "objects": [{"name": "bed", "count": 1}]
  }
}
```

`placement.status` is `direct`, `targeted_retry`, or `unresolved`. A location is
grounded only when it contains a physical relation and anchor. Image coordinates
and generic values such as `center of frame`, `visible scene`, or `office
setting` are rejected. An unresolved placement remains explicit rather than
allowing background objects to masquerade as the target location. The legacy
flat fields remain in each event temporarily so existing artifacts and tools
continue to load.

Reproduce the two event-selection runs on GPUs 4-7:

```bash
cd /home/ZhangHuayu/Workspace/VSRTask
GPUS=4,5,6,7 ./scripts/run_qwen25_memory_qa_ablations.sh
```

Reproduce the isolated location rewrite and its QA run:

```bash
GPUS=4,5,6,7 ./scripts/run_qwen25_physical_where_ablation.sh
```

The QA evaluator also exposes the policies directly through
`--event-score-threshold` and `--event-retrieval {all,entity}`. Each prediction
records source, thresholded, and retrieved event counts plus the extracted query
entity and any retrieval fallback.

## Legacy Full-Delta Training Task

The active task is one small temporal K/V delta predictor. It predicts the
full `delta_K` and `delta_V` tensors for the current frame:

> Which patches in the current frame are edited?

For each known edited frame at index `t`, one causal window is constructed:

```text
[t-7, t-6, ..., t-1, edited frame t]
```

The previous seven frames are temporal context. If another edited frame is
inside the history, its edited PNG is used; otherwise the clean extracted
frame is used.

The frozen Qwen2.5-VL backbone extracts K/V features for all eight frames.
This extraction is performed once by `scripts/cache_vsr_kv_features.py`;
training then reads the saved tensors and does not load Qwen:

```text
[8, 28, 391, 512] K
[8, 28, 391, 512] V
```

The detector then:

```text
K window -> independent two-layer temporal Transformer
V window -> independent two-layer temporal Transformer
           -> current-frame delta tensors [28, 391, 512]
```

Only the last/current frame is supervised. The output is:

```text
[28, 391, 512] predicted_delta_K
[28, 391, 512] predicted_delta_V
```

The final dimension is the complete KV feature vector. K/V and the 28 Qwen
layers are not fused during training.

## Labels

The stored `delta_K` and `delta_V` for the current edited frame are used
directly as regression targets:

```text
target_delta_K: [28, 391, 512]
target_delta_V: [28, 391, 512]
```

The detector directly regresses these tensors with MSE:

```text
loss_K = MSE(predicted_delta_K, target_delta_K)
loss_V = MSE(predicted_delta_V, target_delta_V)
loss = 0.5 * (loss_K + loss_V)
```

For diagnostics only, predicted full deltas can be reduced with RMS to obtain
a patch-level diff score. This reduction is not the training target.

At inference time, the predicted layer diff can be fused into one patch diff:

```text
predicted_delta_K: [28, 391, 512]
predicted_delta_V: [28, 391, 512]
patch_diff = RMS(predicted_delta_K, predicted_delta_V)  # [28, 391]
patch_diff = mean(patch_diff, dim=0)                    # [391]
```

This scoring is post-processing, not a trainable fusion layer.

## Network And Objective

The frozen Qwen backbone is not trained. It extracts K/V features for every
frame in an 8-frame causal window. The trainable detector has this structure:

```text
K: [B, 8, 28, 391, 512] -> K branch -> predicted_delta_K: [B, 28, 391, 512]
V: [B, 8, 28, 391, 512] -> V branch -> predicted_delta_V: [B, 28, 391, 512]
```

Only the last frame is supervised. The target for an edited current frame is
the saved full K/V delta tensors. The target for a clean current frame is an
all-zero delta tensor:

```text
edited window: MSE(predicted_delta_K, delta_K) + MSE(predicted_delta_V, delta_V)
clean window:  MSE(predicted_delta_K, 0) + MSE(predicted_delta_V, 0)
```

The four-card DDP run has one process on each GPU. During detector training,
only the cached K/V tensors and the two detector branches are loaded; Qwen is
not loaded and no Qwen forward pass is performed.

## Saved Training Files

The output directory contains:

```text
run_config.json       command-line arguments and data paths
model_summary.json    network structure, tensor shapes, and parameter counts
train_log.json        per-window logs plus global epoch mean loss
checkpoint-*.pt       detector weights, configuration, and epoch statistics
```

`train_log.json` has this structure:

```json
{
  "window_logs": [
    {
      "epoch": 0,
      "window_type": "edited",
      "loss": 0.0,
      "target_diff_mean": 0.0,
      "predicted_diff_mean": 0.0
    }
  ],
  "epoch_metrics": [
    {
      "epoch": 0,
      "global_windows": 100,
      "global_mean_loss": 0.0
    }
  ]
}
```

## Precompute K/V Features

The precompute stage writes one file per sample:

```text
/data1/ZhangHuayu/datasets/VSR-Train/kv_features/00000001.pt
```

Each file contains the unique frames needed by the positive and sampled
negative windows:

```text
frames[i].k: [28, 391, 512]
frames[i].v: [28, 391, 512]
```

The cached tensors are stored as `float16` by default. Use the same
`--cache-epochs`, `--window-size`, `--negative-windows`, and `--seed` values
as the later training command.

```bash
cd /home/ZhangHuayu/Workspace/VSRTask

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/cache_vsr_kv_features.py \
  --model /data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct \
  --kv-diff-root /data1/ZhangHuayu/datasets/VSR-Train/kv_diff \
  --frames-root /data1/ZhangHuayu/tmp/vsr_train_frames \
  --manifest /data1/ZhangHuayu/datasets/VSR-Train/manifest.json \
  --dataset-root /data1/ZhangHuayu/datasets/VSR-Train \
  --output-root /data1/ZhangHuayu/datasets/VSR-Train/kv_features \
  --sample-range 1 300 \
  --window-size 8 \
  --negative-windows 4 \
  --cache-epochs 10 \
  --dtype float16 \
  --seed 42
```

After caching, train only the two small K/V networks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_patch_detector.py \
  --kv-diff-root /data1/ZhangHuayu/datasets/VSR-Train/kv_diff \
  --kv-cache-root /data1/ZhangHuayu/datasets/VSR-Train/kv_features \
  --frames-root /data1/ZhangHuayu/tmp/vsr_train_frames \
  --output-dir /data1/ZhangHuayu/checkpoints/vsr_full_delta_epoch10 \
  --sample-range 1 300 \
  --window-size 8 \
  --negative-windows 4 \
  --batch-size 4 \
  --detector-d-model 256 \
  --detector-layers 2 \
  --detector-heads 8 \
  --epochs 10 \
  --seed 42 \
  --progress-every 1 \
  --log-every 100
```

In four-card training, `global_mean_loss` is averaged over all windows from
all four processes. The window-level rows are written by rank 0 and are a
process-local trace; use `epoch_metrics` to compare epochs.

The training set contains two types of windows:

```text
positive window: [previous frames, current edited frame]
negative window: [previous clean frames, current clean frame]
```

Each edited frame creates one positive window. Clean negative windows are
sampled from positions whose current frame and history do not include any
known edited frame. Their target is two all-zero delta tensors. By default,
4 clean negative windows are sampled per video, while the rest of the clean
video is not enumerated.

`train_diff_transformer.py` remains a legacy alternative implementation. The
active full-delta path is `train_patch_detector.py`.

## Direct Training Command

```bash
cd /home/ZhangHuayu/Workspace/VSRTask

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ZhangHuayu/Workspace/VSRTask \
/home/ZhangHuayu/miniconda3/envs/vsr_train/bin/torchrun \
--standalone --nproc_per_node=4 \
scripts/train_patch_detector.py \
  --kv-diff-root /data1/ZhangHuayu/datasets/VSR-Train/kv_diff \
  --kv-cache-root /data1/ZhangHuayu/datasets/VSR-Train/kv_features \
  --frames-root /data1/ZhangHuayu/tmp/vsr_train_frames \
  --output-dir /data1/ZhangHuayu/checkpoints/vsr_full_delta_epoch10 \
  --window-size 8 \
  --negative-windows 4 \
  --detector-layers 2 \
  --epochs 10 \
  --seed 42 \
  --log-every 100
```

This launches four DDP processes, one on each of GPUs `0,1,2,3`. The Qwen
forward pass has already been completed by the cache stage. The checkpoint
and log are written only by rank 0.

The four edited PNGs referenced by each diff record must exist. The clean
frames under `--frames-root` must use the same FPS and frame indexing as the
video used to generate the diff records.
