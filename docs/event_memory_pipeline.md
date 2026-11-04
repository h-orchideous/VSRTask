# Event Memory VSR Pipeline

This pipeline keeps anomaly detection, semantic memory writing, and question
answering as separate stages.

The Qwen2.5-VL and Qwen3.5 experiments use separate entry files. They do not
select a model implementation automatically:

- `run_event_memory_pipeline_qwen25.sh` uses Qwen2.5-VL for detection,
  semantic writing, and QA.
- `run_event_memory_pipeline.sh` retains the Qwen3.5 semantic writer and QA.

## Architecture

```text
video stream
  -> frozen Qwen2.5-VL K/V features
  -> PatchDiff detector
  -> threshold-triggered event images
  -> Qwen2.5-VL or Qwen3.5 query-independent semantic writer
  -> ordered explicit event memory
  -> question-aware QA using the same writer version
```

The VSR question is not available during detection or semantic memory writing.
It is introduced only during the final QA stage.

At QA time, events can optionally be filtered by detector score or retrieved by
the entity named in the question. This is query-aware reading of an already
written memory; it does not expose the question to detection or writing. The
selected events always remain ordered by `peak_frame`.

The semantic writer receives the original selected frame and a clean crop from
the original pixels around the highest-scoring PatchDiff token. The red overlay
is retained only as detector audit evidence and is not sent to the semantic
model. The crop is an attention cue, not the memory itself. New records separate
`anomaly`, `placement`, and `context`. A physical placement must contain both a
relation and visible anchor; frame coordinates and generic scene labels are
marked unresolved. Legacy flat fields remain during migration. This supports
temporal ordering, object properties, and counting without rewriting memory for
a particular question.

## Memory Record

Each frame whose anomaly score is greater than or equal to the configured
threshold remains an independent event. Temporal merging and Top-K selection
are disabled because adjacent inserted frames can represent different ordered
events and the event count is unknown at inference time. There is no default
event-count cap; `MAX_MEMORY_EVENTS` is only an optional resource safeguard.
New runs use `configs/event_threshold_balanced.json` by default. The recorded
legacy baseline keeps its original threshold in its output metadata.

```json
{
  "event_id": 1,
  "peak_frame": 43,
  "peak_seconds": 43.0,
  "peak_score": 1.29,
  "episodic_memory": {
    "semantic_schema_version": 3,
    "what": "a stationary golden retriever toy",
    "where": "on a round table",
    "scene": "kitchen",
    "background": "round table, chairs, and cabinets",
    "objects": [
      {"name": "golden retriever toy", "count": 1},
      {"name": "table", "count": 1},
      {"name": "chair", "count": 2}
    ],
    "anomaly": {
      "entity": "golden retriever toy",
      "description": "a stationary golden retriever toy",
      "count": 1
    },
    "placement": {
      "relation": "on",
      "anchor": "a round table",
      "text": "on a round table",
      "status": "direct",
      "observation_text": "on a round table"
    },
    "context": {
      "scene": "kitchen",
      "summary": "round table, chairs, and cabinets",
      "objects": [
        {"name": "table", "count": 1},
        {"name": "chair", "count": 2}
      ]
    },
    "semantic_writer": {
      "query_conditioned": false
    }
  }
}
```

## Run

Run the Qwen2.5-VL baseline first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/run_event_memory_pipeline_qwen25.sh all
```

Its default output root is
`outputs/vsi_super_recall_event_memory_pipeline_qwen25`.

Run the Qwen3.5 variant separately:

Run all three stages:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/run_event_memory_pipeline.sh all
```

Run one stage or resume after a completed stage:

```bash
./scripts/run_event_memory_pipeline_qwen25.sh detect
./scripts/run_event_memory_pipeline_qwen25.sh caption
./scripts/run_event_memory_pipeline_qwen25.sh qa

# Qwen3.5 variant
./scripts/run_event_memory_pipeline.sh detect
./scripts/run_event_memory_pipeline.sh caption
./scripts/run_event_memory_pipeline.sh qa
```

Select a sample range:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
SAMPLE_START=301 SAMPLE_END=301 \
DATASET_ROOT=/data1/ZhangHuayu/datasets/VSR-Train \
SOURCE_PARQUET=/home/ZhangHuayu/Workspace/VSRTask/outputs/vsr_gt_top4_301_350/test_10mins_301_350.parquet \
RUN_ROOT=/home/ZhangHuayu/Workspace/VSRTask/outputs/event_memory_pipeline_validation_301 \
./scripts/run_event_memory_pipeline.sh all
```

## Outputs

```text
RUN_ROOT/
  detection/
    memories/
    evidence/
    frame_scores/
    memory_manifest.json
  semantic/
    memories/
    semantic_memory_manifest.json
  qa/
    predictions.rank*.jsonl
    results.json
  validation_report.json
```

## Validation Metrics

1. Detector: edited-frame recall, frame AP, and false-positive event count.
2. Writer: anomaly entity, location, count, and relation accuracy.
3. QA: VSR multiple-choice accuracy.
4. Ablation: detector Top-4 images versus explicit semantic memory.

Sparse selected frames must be passed as ordered images. Re-encoding unrelated
selected frames as a short video invokes Qwen2.5-VL temporal tubelet fusion and
was measured to reduce the oracle four-frame accuracy from 90% to 38%.

## Streaming GPU Pipeline

The streaming launcher assigns persistent detector and semantic workers to
different GPU groups. A detector enqueues an event immediately when the current
frame score crosses the threshold. Semantic workers atomically claim events and
write one durable memory JSON per event while detection continues.

```bash
DETECT_GPUS=0,1 \
SEMANTIC_GPUS=2,3 \
DURATION=10mins \
./scripts/run_stream_event_memory_qwen25.sh all
```

`DURATION` selects the official VSI-SUPER-Recall split and supports `10mins`,
`30mins`, `60mins`, `120mins`, and `240mins`. The corresponding parquet is
resolved as `/data1/ZhangHuayu/datasets/VSI-SUPER-Recall/test_${DURATION}.parquet`.

Use `stream` instead of `all` to stop after detection, semantic writing, and
ordered memory assembly without running QA. The queue directories are
restartable: unfinished `processing` jobs are returned to `pending`, and
completed event files are not generated twice.

## QA Event Selection

The streaming launcher accepts these environment variables for QA-only runs:

```bash
QA_EVENT_SCORE_THRESHOLD=0.9073798060417175 \
QA_EVENT_RETRIEVAL=all \
QA_OUTPUT_DIR="$RUN_ROOT/qa_old_threshold" \
./scripts/run_stream_event_memory_qwen25.sh qa

QA_EVENT_RETRIEVAL=entity \
QA_OUTPUT_DIR="$RUN_ROOT/qa_entity_retrieval" \
./scripts/run_stream_event_memory_qwen25.sh qa
```

`entity` retrieval extracts the target from the VSR question, matches it only
against `what`, `anomaly_entity`, and the visible-object inventory, and preserves
time order. It does not inspect answer options. If no semantic event matches,
the evaluator falls back to the score-filtered memory and records the fallback.
