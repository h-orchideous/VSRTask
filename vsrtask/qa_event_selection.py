from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


QUESTION_ENTITY_PATTERNS = (
    re.compile(
        r"order\s+in\s+which\s+the\s+(.+?)\s+appeared\s+in\s+the\s+video",
        re.IGNORECASE,
    ),
    re.compile(r"order\s+of\s+(?:the\s+)?(.+?)(?:\?|$)", re.IGNORECASE),
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "appeared",
    "in",
    "of",
    "the",
    "video",
}
GENERIC_PREFIXES = ("bouquet of ", "picture of ", "image of ")
ENTITY_ALIASES = {
    "bouquet of red roses": (
        "red roses",
        "red rose",
        "roses",
        "rose",
        "bouquet",
        "bouquets",
        "flowers",
    ),
    "golden retriever": ("golden retriever", "retriever", "dog"),
    "statue of liberty": ("statue of liberty", "liberty statue", "statue"),
    "teddy bear": ("teddy bear", "teddy", "bear"),
    "ultraman": ("ultraman", "ultrman"),
    "white ragdoll cat": (
        "white ragdoll cat",
        "ragdoll cat",
        "rag doll cat",
        "ragdoll",
        "cat",
    ),
    "yellow parrot": ("yellow parrot", "parrot", "parrots"),
}


@dataclass(frozen=True)
class EventSelection:
    events: list[dict[str, Any]]
    source_event_count: int
    threshold_event_count: int
    retrieved_event_count: int
    query_entity: str | None
    retrieval_fallback: str | None


def normalize_text(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value or "").lower()))


def normalized_tokens(value: Any) -> set[str]:
    tokens = set(TOKEN_RE.findall(str(value or "").lower())) - STOP_WORDS
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def extract_query_entity(question: str) -> str | None:
    text = " ".join(str(question).split())
    for pattern in QUESTION_ENTITY_PATTERNS:
        match = pattern.search(text)
        if match:
            entity = match.group(1).strip(" .,:;?!\"'")
            return entity or None
    return None


def entity_aliases(entity: str) -> set[str]:
    normalized = normalize_text(entity)
    aliases = {normalized}
    aliases.update(ENTITY_ALIASES.get(normalized, ()))
    for prefix in GENERIC_PREFIXES:
        if normalized.startswith(prefix):
            aliases.add(normalized[len(prefix) :])
    return {normalize_text(alias) for alias in aliases if alias}


def event_entity_text(event: dict[str, Any]) -> str:
    episodic = event.get("episodic_memory") or {}
    anomaly = episodic.get("anomaly") or {}
    values = [
        anomaly.get("entity"),
        anomaly.get("description"),
        episodic.get("what"),
        episodic.get("anomaly_entity"),
    ]
    objects = episodic.get("objects") or episodic.get("entities") or []
    values.extend(item.get("name") for item in objects if isinstance(item, dict))
    return normalize_text(" ".join(str(value or "") for value in values))


def event_matches_entity(event: dict[str, Any], entity: str) -> bool:
    semantic_text = event_entity_text(event)
    if not semantic_text:
        return False
    semantic_tokens = normalized_tokens(semantic_text)
    for alias in entity_aliases(entity):
        if alias and alias in semantic_text:
            return True
        alias_tokens = normalized_tokens(alias)
        if alias_tokens and alias_tokens.issubset(semantic_tokens):
            return True
    return False


def select_events(
    memory: dict[str, Any],
    question: str,
    *,
    score_threshold: float | None = None,
    retrieval: str = "all",
) -> EventSelection:
    if retrieval not in {"all", "entity"}:
        raise ValueError(f"Unsupported event retrieval mode: {retrieval}")

    source_events = sorted(
        memory.get("events", []), key=lambda item: int(item["peak_frame"])
    )
    threshold_events = [
        event
        for event in source_events
        if score_threshold is None
        or float(event.get("peak_score", float("-inf"))) >= score_threshold
    ]
    query_entity = extract_query_entity(question) if retrieval == "entity" else None
    fallback = None
    selected = threshold_events
    if retrieval == "entity":
        if query_entity is None:
            fallback = "query_entity_not_found_use_threshold_events"
        else:
            matches = [
                event
                for event in threshold_events
                if event_matches_entity(event, query_entity)
            ]
            if matches:
                selected = matches
            else:
                fallback = "no_entity_match_use_threshold_events"

    return EventSelection(
        events=selected,
        source_event_count=len(source_events),
        threshold_event_count=len(threshold_events),
        retrieved_event_count=len(selected),
        query_entity=query_entity,
        retrieval_fallback=fallback,
    )
