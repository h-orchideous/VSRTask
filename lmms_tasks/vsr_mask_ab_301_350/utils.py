import os
import re
from collections import OrderedDict

import datasets
from loguru import logger as eval_logger


_SAMPLE_ID_RE = re.compile(r"(\d{8})$")


def _sample_id(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    match = _SAMPLE_ID_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot extract sample id from video path: {video_path}")
    return int(match.group(1))


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    start = int(os.getenv("VSR_SAMPLE_START", "301"))
    end = int(os.getenv("VSR_SAMPLE_END", "350"))
    filtered = dataset.filter(lambda doc: start <= _sample_id(doc["video_path"]) <= end)
    eval_logger.info(f"[vsr_mask_ab] sample_range={start}-{end} count={len(filtered)}")
    return filtered


def doc_to_visual(doc):
    root = os.environ["VSI_SUPER_RECALL_ROOT"]
    video_path = doc["video_path"]
    if not os.path.isabs(video_path):
        video_path = os.path.join(root, video_path)
    video_path = os.path.normpath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Missing evaluation video: {video_path}")
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    options = "\n".join(doc["options"])
    return (
        f"{doc['question'].strip()}\nOptions:\n{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def _normalized_choice(prediction):
    return prediction.split(" ")[0].rstrip(".").strip().upper()


def process_results(doc, results):
    prediction = results[0]
    sample_id = _sample_id(doc["video_path"])
    result = {
        "sample_id": sample_id,
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

    sample_ids = sorted(doc["sample_id"] for doc in docs)
    evaluated_range = (
        f"{sample_ids[0]}-{sample_ids[-1]}" if sample_ids else "empty"
    )

    independent = [doc for doc in docs if doc["sample_id"] >= 311]
    independent_correct = sum(doc["accuracy"] for doc in independent)
    independent_accuracy = (
        100.0 * independent_correct / len(independent) if independent else 0.0
    )

    eval_logger.info(f"Evaluated accuracy ({evaluated_range}): {accuracy:.3f}")
    eval_logger.info(f"Independent accuracy (311-350): {independent_accuracy:.3f}")

    outputs = OrderedDict()
    outputs[f"Evaluated_{evaluated_range}"] = accuracy
    outputs["Independent_311_350"] = independent_accuracy
    outputs["Correct"] = int(correct)
    outputs["Total"] = total
    return outputs
