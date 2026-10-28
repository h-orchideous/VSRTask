import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import datasets
from loguru import logger as eval_logger
from PIL import Image


_SAMPLE_ID_RE = re.compile(r"(\d{8})$")
_REPORT_CACHE = None


def _sample_id(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    match = _SAMPLE_ID_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot extract sample id from video path: {video_path}")
    return int(match.group(1))


def _frame_map():
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        report_path = Path(os.environ["VSR_GT_TOP4_BUILD_REPORT"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _REPORT_CACHE = {
            record["video_path"]: record["edited_frames_temporal_order"]
            for record in report["records"]
        }
    return _REPORT_CACHE


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    start = int(os.getenv("VSR_SAMPLE_START", "301"))
    end = int(os.getenv("VSR_SAMPLE_END", "350"))
    filtered = dataset.filter(lambda doc: start <= _sample_id(doc["video_path"]) <= end)
    eval_logger.info(
        f"[vsr_gt_top4_images] sample_range={start}-{end} count={len(filtered)}"
    )
    return filtered


def doc_to_visual(doc):
    paths = _frame_map().get(doc["video_path"])
    if paths is None:
        raise KeyError(f"No edited-frame record for {doc['video_path']}")
    images = []
    for path_value in paths:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing edited frame: {path}")
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    if len(images) != 4:
        raise ValueError(f"Expected four images for {doc['video_path']}, got {len(images)}")
    return images


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    options = "\n".join(doc["options"])
    return (
        f"{doc['question'].strip()}\nOptions:\n{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def _normalized_choice(prediction):
    match = re.search(r"\b([A-D])\b", prediction.upper())
    return match.group(1) if match else ""


def process_results(doc, results):
    prediction = results[0]
    result = {
        "sample_id": _sample_id(doc["video_path"]),
        "video_path": doc["video_path"],
        "prediction": prediction,
        "answer": doc["answer"],
        "accuracy": float(_normalized_choice(prediction) == doc["answer"].upper()),
    }
    return {"score": result}


def aggregate_results(docs):
    total = len(docs)
    correct = sum(doc["accuracy"] for doc in docs)
    accuracy = 100.0 * correct / total if total else 0.0
    independent = [doc for doc in docs if doc["sample_id"] >= 311]
    independent_correct = sum(doc["accuracy"] for doc in independent)
    independent_accuracy = (
        100.0 * independent_correct / len(independent) if independent else 0.0
    )
    outputs = OrderedDict()
    outputs["Evaluated_301-350"] = accuracy
    outputs["Independent_311-350"] = independent_accuracy
    outputs["Correct"] = int(correct)
    outputs["Total"] = total
    eval_logger.info(f"Direct four-image accuracy: {accuracy:.3f}")
    eval_logger.info(f"Independent 311-350 accuracy: {independent_accuracy:.3f}")
    return outputs
