from __future__ import annotations

import re
from typing import Any


SEMANTIC_TEXT_REJECT = re.compile(
    r"addCriterion|matchCondition|(?-i:add[A-Z])|<\||自动生成|请描述|请提供|作为(?:一个|AI)|"
    r"language model|system prompt|assistant response",
    re.IGNORECASE,
)
OVERLAY_ARTIFACT_REJECT = re.compile(
    r"\b(?:patch\s*diff|overlay|bounding boxes?|detector[- ]selected "
    r"(?:patch(?:es)?|region(?:s)?)|highlighted (?:patch(?:es)?|region(?:s)?|"
    r"area(?:s)?|box(?:es)?)|red (?:square(?:s)?|rectangle(?:s)?|patch(?:es)?|"
    r"marker(?:s)?|outline(?:s)?|box(?:es)?)|square marker(?:s)?|"
    r"grid (?:cell(?:s)?|square(?:s)?))\b",
    re.IGNORECASE,
)
OBSERVATION_LABELS = (
    "What",
    "Where",
    "Scene",
    "Background",
)
OBSERVATION_LABEL_PATTERN = re.compile(
    rf"(?:{'|'.join(OBSERVATION_LABELS)}):",
    re.IGNORECASE,
)
FLEXIBLE_OBSERVATION_LABEL_RE = re.compile(
    r"(?P<label>what|(?:unusual\s+)?foreground\s+object|"
    r"where(?:\s+(?:on|in|at))?|scene(?:\s+(?:in|type))?|"
    r"background(?:\s+includes?)?)\s*:",
    re.IGNORECASE,
)
FRAME_COORDINATE_RE = re.compile(
    r"\b(?:center|centre|left|right|top|bottom|middle)\s+(?:of\s+)?(?:the\s+)?"
    r"(?:frame|image|picture|crop)\b",
    re.IGNORECASE,
)
ANOMALY_DESCRIPTION_REJECT = re.compile(
    r"(?:\*{2,}|!{2,}|\bconcise(?:ly|ness|ous)?\b|"
    r"\bvisible\s+(?:attributes?|state)\s*:|\banomal(?:y|ous)\s+object\s*:)",
    re.IGNORECASE,
)
ANOMALY_CROSS_FIELD_REJECT = re.compile(
    r"(?:\b(?:where|scene|background|addcriterion|matchcondition)\b|"
    r"\bforeground(?:\s+object)?\s*:)",
    re.IGNORECASE,
)
PHYSICAL_RELATION_RE = re.compile(
    r"^\s*(?P<relation>in front of|on top of|next to|inside|within|on|in|near|"
    r"beside|under|above|against|atop|behind|by)\s+"
    r"(?P<anchor>.+?)\s*$",
    re.IGNORECASE,
)
EMBEDDED_PHYSICAL_RELATION_RE = re.compile(
    r"\b(?P<relation>in front of|on top of|next to|inside|within|on|in|near|"
    r"beside|under|above|against|atop|behind|by)\s+"
    r"(?P<anchor>[^|;\n]+?)\s*$",
    re.IGNORECASE,
)
VAGUE_PLACEMENT_RE = re.compile(
    r"\b(?:depicted scene|visible scene|office setting|indoor setting|"
    r"outdoor setting|scene|frame|image|picture|foreground|background)\b",
    re.IGNORECASE,
)


def is_degenerate(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 20:
        return False
    unique_ratio = len(set(words)) / len(words)
    most_common = max(words.count(word) for word in set(words))
    return unique_ratio < 0.35 or most_common >= 8


def parse_narrative(text: str) -> dict[str, Any]:
    fields: dict[str, str | None] = {}
    for key in ("Where", "What", "Relations"):
        match = re.search(
            rf"{key}:\s*(.*?)(?=\s*;?\s*(?:Where|What|Relations):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        fields[key.lower()] = match.group(1).strip() if match else None

    # Recover the observed "Where: <place>: <object> Relations: ..." format.
    if not fields.get("what"):
        legacy = re.fullmatch(
            r"\s*Where:\s*([^:\n]+):\s*(.*?)\s+Relations:\s*(.*?)\s*",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if legacy:
            fields = {
                "where": legacy.group(1).strip(),
                "what": legacy.group(2).strip(),
                "relations": legacy.group(3).strip(),
            }
    return fields


def parse_full_frame_observation(text: str) -> dict[str, str | None]:
    """Parse the fixed labels used by the anomaly-centred full-frame writer."""
    fields: dict[str, str | None] = {}
    lookahead = "|".join(OBSERVATION_LABELS)
    for label in OBSERVATION_LABELS:
        match = re.search(
            rf"{label}:\s*(.*?)(?=\s*;?\s*(?:{lookahead}):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        fields[label.lower()] = match.group(1).strip() if match else None
    if not full_frame_observation_complete(fields):
        normalized = re.sub(r"[*`]", "", text)
        matches = list(FLEXIBLE_OBSERVATION_LABEL_RE.finditer(normalized))
        candidates: dict[str, list[str]] = {
            "what": [],
            "where": [],
            "scene": [],
            "background": [],
        }
        for index, match in enumerate(matches):
            label = match.group("label").lower()
            if "foreground object" in label or label == "what":
                canonical = "what"
            elif label.startswith("where"):
                canonical = "where"
            elif label.startswith("scene"):
                canonical = "scene"
            else:
                canonical = "background"
            end = matches[index + 1].start() if index + 1 < len(matches) else None
            value = normalized[match.end() : end].strip(" \t\r\n-:;.")
            value = trim_semantic_contamination(value)
            if value:
                candidates[canonical].append(value)
        flexible: dict[str, str | None] = {}
        for canonical, values in candidates.items():
            validator = (
                anomaly_description_clean
                if canonical == "what"
                else observation_text_clean
            )
            flexible[canonical] = next(
                (value for value in values if validator(value)),
                None,
            )
        if observation_field_score(flexible) > observation_field_score(fields):
            fields = flexible
    if not full_frame_observation_complete(fields):
        parts = [part.strip() for part in re.split(r"\s*\|\|\s*", text)]
        if len(parts) == 4:
            piped = dict(zip(("what", "where", "scene", "background"), parts))
            if observation_field_score(piped) > observation_field_score(fields):
                fields = piped
    return fields


def observation_field_score(fields: dict[str, Any]) -> int:
    """Count independently valid fields in a possibly incomplete observation."""
    score = int(anomaly_description_clean(fields.get("what")))
    score += sum(
        int(observation_text_clean(fields.get(key)))
        for key in ("where", "scene", "background")
    )
    return score


def merge_observation_attempts(
    attempts: list[str],
    *,
    fallback_what: str | None = None,
) -> tuple[dict[str, str], list[str]] | None:
    """Merge valid fields across image-conditioned retries without inventing facts."""
    merged: dict[str, str] = {}
    for attempt in attempts:
        fields = parse_full_frame_observation(attempt)
        if "what" not in merged and anomaly_description_clean(fields.get("what")):
            merged["what"] = str(fields["what"])
        for key in ("where", "scene", "background"):
            if key not in merged and observation_text_clean(fields.get(key)):
                merged[key] = str(fields[key])
    if "what" not in merged and anomaly_description_clean(fallback_what):
        merged["what"] = str(fallback_what)
    if "what" not in merged:
        return None
    missing = [
        key for key in ("where", "scene", "background") if key not in merged
    ]
    for key in missing:
        merged[key] = "unknown"
    return merged, missing


def parse_physical_where_response(text: str) -> str | None:
    """Parse a location-only generation and reject image-coordinate answers."""
    if not isinstance(text, str):
        return None
    value = re.sub(r"^\s*Where\s*:\s*", "", text, flags=re.IGNORECASE)
    if re.search(r"\b(?:What|Scene|Background)\s*:", value, re.IGNORECASE):
        return None
    value = trim_semantic_contamination(value)
    if (
        not semantic_text_clean(value, field="where")
        or FRAME_COORDINATE_RE.search(value or "")
    ):
        return None
    value = re.sub(r"\s+", " ", value or "").strip(" .;,\n")
    placement = parse_grounded_placement(value)
    return placement["text"] if placement else None


def parse_grounded_placement(value: Any) -> dict[str, str] | None:
    """Parse a physical relation and anchor, rejecting scene/frame locations."""
    cleaned = trim_semantic_contamination(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,\n")
    if (
        FRAME_COORDINATE_RE.search(cleaned)
        or VAGUE_PLACEMENT_RE.search(cleaned)
        or not semantic_text_clean(cleaned, field="where")
    ):
        return None
    match = PHYSICAL_RELATION_RE.fullmatch(cleaned)
    if not match:
        return None
    relation = match.group("relation").lower()
    anchor = match.group("anchor").strip(" .;,\n")
    if not anchor or VAGUE_PLACEMENT_RE.fullmatch(anchor):
        return None
    return {
        "relation": relation,
        "anchor": anchor,
        "text": f"{relation} {anchor}",
    }


def build_structured_event_memory(
    fields: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    targeted_where: str | None = None,
) -> dict[str, Any]:
    """Build the canonical v3 anomaly/placement/context event structure."""
    anomaly_object = objects[0] if objects else {}
    direct = parse_grounded_placement(fields.get("where"))
    targeted = parse_grounded_placement(targeted_where)
    if targeted is not None:
        placement = {**targeted, "status": "targeted_retry"}
    elif direct is not None:
        placement = {**direct, "status": "direct"}
    else:
        placement = {
            "relation": None,
            "anchor": None,
            "text": "unknown",
            "status": "unresolved",
        }
    placement["observation_text"] = fields.get("where")

    return {
        "anomaly": {
            "entity": entity_from_what(fields.get("what")),
            "description": fields.get("what"),
            "count": anomaly_object.get("count", 1),
        },
        "placement": placement,
        "context": {
            "scene": fields.get("scene"),
            "summary": fields.get("background"),
            "objects": objects[1:],
        },
    }


SCENE_PATTERN = re.compile(
    r"\b(?:living room|bedroom|kitchen|office|bathroom|dining room|hallway|"
    r"classroom|workspace|garage|store|shop|outdoor(?: scene| setting)?|"
    r"indoor(?: scene| setting| interior| room| space)?)\b",
    re.IGNORECASE,
)
LOCATION_PATTERN = re.compile(
    r"\b(?:on|in|inside|near|beside|under|above|atop|against|next to)\b.+",
    re.IGNORECASE,
)


def recover_unstructured_observation(text: str) -> dict[str, str | None] | None:
    """Recover the four memory fields from a clean but unlabeled observation."""
    clean = trim_semantic_contamination(text)
    if (
        not clean
        or OVERLAY_ARTIFACT_REJECT.search(clean)
        or "<" in clean
        or ">" in clean
    ):
        return None
    clean = re.sub(r"\s+", " ", clean).strip(" .;,")
    if not clean:
        return None

    # Some generations include useful values but omit or misplace the labels.
    # Treat those labels as separators before applying the conservative fallback.
    partition_text = re.sub(
        r"\b(?:What|Where|Scene|Background):\s*",
        "; ",
        clean,
        flags=re.IGNORECASE,
    ).strip(" ;")

    semicolon_parts = [
        part.strip(" .;,")
        for part in re.split(r"[;\n]+", partition_text)
        if part.strip()
    ]
    comma_parts = [
        part.strip(" .;,") for part in partition_text.split(",") if part.strip()
    ]
    parts = semicolon_parts if len(semicolon_parts) > 1 else comma_parts
    what = parts[0] if parts else clean
    if len(what) < 3:
        return None

    location_match = LOCATION_PATTERN.search(what)
    where = location_match.group(0).strip(" .;,\n") if location_match else None
    scene_match = SCENE_PATTERN.search(clean)
    scene = scene_match.group(0).strip() if scene_match else None

    remaining = [part for part in parts[1:] if part not in {where, scene}]
    background = ", ".join(remaining).strip() if remaining else None
    if where is None:
        where = "visible in the depicted scene"
    if scene is None:
        scene = "indoor scene"
    if background is None:
        background = clean

    def concise(value: str, limit: int = 220) -> str:
        if len(value) <= limit:
            return value
        shortened = value[:limit].rsplit(" ", 1)[0].rstrip(" .;,")
        return shortened or value[:limit]

    recovered = {
        "what": concise(what),
        "where": concise(where),
        "scene": concise(scene),
        "background": concise(background),
    }
    return recovered if full_frame_observation_complete(recovered) else None


def trim_semantic_contamination(value: Any) -> str | None:
    """Keep valid leading content when a generation appends prompt artifacts."""
    if not isinstance(value, str):
        return None
    match = SEMANTIC_TEXT_REJECT.search(value)
    text = value[: match.start()] if match else value
    text = text.strip(" \t\r\n;,:-<>")
    return text or None


def semantic_text_clean(value: Any, *, field: str) -> bool:
    """Reject non-empty strings that contain prompt/template contamination."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if (
        not text
        or SEMANTIC_TEXT_REJECT.search(text)
        or OVERLAY_ARTIFACT_REJECT.search(text)
        or "<" in text
        or ">" in text
    ):
        return False
    if field in {"where", "what"} and re.search(
        r"(?:Where|What|Relations):", text, re.IGNORECASE
    ):
        return False
    if len(text) > 240 or text.count("\n") > 2 or is_degenerate(text):
        return False
    return True


def observation_text_clean(value: Any) -> bool:
    """Validate one field from a structured full-frame observation."""
    return semantic_text_clean(value, field="observation") and not (
        isinstance(value, str)
        and (
            OBSERVATION_LABEL_PATTERN.search(value)
            or re.search(r"\bRelations\s*:", value, re.IGNORECASE)
        )
    )


def anomaly_description_clean(value: Any) -> bool:
    """Reject empty/template-like anomaly descriptions before memory writing."""
    if not observation_text_clean(value):
        return False
    text = re.sub(r"\s+", " ", str(value)).strip(" .,:;\"'")
    if not text or re.fullmatch(r"[A-D]", text, re.IGNORECASE):
        return False
    if "|" in text:
        return False
    if len(re.sub(r"[^a-z0-9]", "", text.lower())) < 3:
        return False
    return (
        ANOMALY_DESCRIPTION_REJECT.search(text) is None
        and ANOMALY_CROSS_FIELD_REJECT.search(text) is None
    )


def parse_anomaly_response(value: Any) -> str | None:
    """Normalize a dedicated anomaly-only generation into one noun phrase."""
    if not isinstance(value, str):
        return None
    cleaned = re.sub(
        r"\b(?:addCriterion|matchCondition)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = trim_semantic_contamination(cleaned)
    if not cleaned:
        return None
    cleaned = re.sub(r"[*`]", "", cleaned).strip()
    cleaned = re.sub(
        r"^(?:What|(?:Unusual\s+)?Foreground\s+Object)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if "||" in cleaned:
        cleaned = cleaned.split("||", 1)[0]
    # Some generations split a valid noun phrase around an injected control
    # token (for example, ``pl\naddCriterion\nplastic bag``). After removing
    # the token, retain the remaining words instead of taking only line one.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[A-D]\s*:\s*(?=\w)", "", cleaned).strip(" .,:;-\n")
    if " - " in cleaned:
        cleaned = cleaned.split(" - ", 1)[0].strip()
    embedded = EMBEDDED_PHYSICAL_RELATION_RE.search(cleaned)
    if embedded:
        cleaned = cleaned[: embedded.start()].strip(" .,:;-\n")
    tokens = cleaned.split()
    if len(tokens) > 1 and len(re.sub(r"[^a-z0-9]", "", tokens[0].lower())) <= 2:
        cleaned = " ".join(tokens[1:])
    return cleaned if anomaly_description_clean(cleaned) else None


def choose_anomaly_description(
    dedicated: str,
    observation: Any,
) -> str:
    """Prefer an observation identity only when the dedicated name is generic."""
    dedicated_clean = parse_anomaly_response(dedicated) or dedicated
    observation_clean = parse_anomaly_response(observation)
    generic = {
        "object",
        "toy",
        "stuffed toy",
        "plush toy",
        "figurine",
        "poster",
        "flowers",
        "dog",
        "cat",
    }
    if (
        normalize_entity_phrase(dedicated_clean) in generic
        and observation_clean
        and len(observation_clean.split()) > len(dedicated_clean.split())
    ):
        return observation_clean
    return dedicated_clean


def normalize_entity_phrase(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip(" .,:;-\n")


def embedded_grounded_placement(value: Any) -> dict[str, str] | None:
    """Extract a trailing physical relation from a dedicated anomaly response."""
    cleaned = trim_semantic_contamination(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-\n")
    match = EMBEDDED_PHYSICAL_RELATION_RE.search(cleaned)
    if not match:
        return None
    return parse_grounded_placement(
        f"{match.group('relation')} {match.group('anchor')}"
    )


def choose_placement_text(
    observation_where: Any,
    anomaly_response: Any,
    targeted_where: Any,
) -> str | None:
    """Choose grounded placement using agreement with full-frame evidence."""
    direct = parse_grounded_placement(observation_where)
    embedded = embedded_grounded_placement(anomaly_response)
    targeted = parse_grounded_placement(targeted_where)
    observation_tokens = normalized_tokens(observation_where)

    if embedded and observation_tokens & normalized_tokens(embedded["anchor"]):
        return embedded["text"]
    if direct:
        return direct["text"]
    if embedded:
        return embedded["text"]
    if targeted:
        return targeted["text"]
    return None


def normalized_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower())) - {
        "a",
        "an",
        "the",
    }


def full_frame_observation_complete(memory: dict[str, Any]) -> bool:
    """Validate the compact v2 anomaly and full-frame fields."""
    required = (
        memory.get("where"),
        memory.get("scene"),
        memory.get("background"),
    )
    return anomaly_description_clean(memory.get("what")) and all(
        observation_text_clean(value) for value in required
    )


def object_inventory_complete(memory: dict[str, Any]) -> bool:
    """Validate the visible-object inventory written with a v2 observation."""
    objects = memory.get("objects")
    if not isinstance(objects, list) or not objects:
        return False
    for item in objects:
        if not isinstance(item, dict):
            return False
        if not semantic_text_clean(item.get("name"), field="entity"):
            return False
        if not isinstance(item.get("count"), int) or item["count"] <= 0:
            return False
    return True


def semantic_memory_complete(memory: dict[str, Any] | None) -> bool:
    """Return whether an event is structurally complete and semantically clean."""
    if not isinstance(memory, dict):
        return False
    schema_version = int(memory.get("semantic_schema_version", 1))
    if schema_version >= 3:
        anomaly = memory.get("anomaly") or {}
        placement = memory.get("placement") or {}
        context = memory.get("context") or {}
        status = placement.get("status")
        placement_complete = (
            status == "unresolved"
            and placement.get("text") == "unknown"
            and observation_text_clean(placement.get("observation_text"))
        ) or (
            status in {"direct", "targeted_retry"}
            and semantic_text_clean(placement.get("relation"), field="relation")
            and semantic_text_clean(placement.get("anchor"), field="anchor")
            and semantic_text_clean(placement.get("text"), field="where")
            and observation_text_clean(placement.get("observation_text"))
        )
        structured_complete = (
            semantic_text_clean(anomaly.get("entity"), field="entity")
            and anomaly_description_clean(anomaly.get("description"))
            and isinstance(anomaly.get("count"), int)
            and anomaly["count"] > 0
            and placement_complete
            and observation_text_clean(context.get("scene"))
            and observation_text_clean(context.get("summary"))
            and isinstance(context.get("objects"), list)
        )
        return (
            structured_complete
            and full_frame_observation_complete(memory)
            and object_inventory_complete(memory)
        )
    if schema_version >= 2:
        return (
            full_frame_observation_complete(memory)
            and object_inventory_complete(memory)
        )
    entity = memory.get("anomaly_entity")
    count = memory.get("anomaly_count")
    legacy_complete = (
        semantic_text_clean(memory.get("where"), field="where")
        and semantic_text_clean(memory.get("what"), field="what")
        and semantic_text_clean(entity, field="entity")
        and isinstance(count, int)
        and count > 0
    )
    return legacy_complete


def entity_from_what(value: str | None) -> str:
    """Derive a stable entity label when count generation is malformed."""
    text = re.sub(r"\s+", " ", (value or "").strip().lower())
    text = re.split(
        r"\s+(?:on|in|inside|near|beside|under|above|atop|against)\s+",
        text,
        maxsplit=1,
    )[0]
    text = text.strip(" .,:;\"'")
    return text or "unknown foreground object"


def entity_matches_what(entity: str, what: str | None) -> bool:
    ignored = {"a", "an", "the", "of", "for", "and", "with"}
    entity_words = set(re.findall(r"[a-z0-9]+", entity.lower())) - ignored
    what_words = set(re.findall(r"[a-z0-9]+", (what or "").lower())) - ignored
    return bool(entity_words & what_words)


def repair_semantic_memory(memory: dict[str, Any] | None) -> dict[str, Any] | None:
    """Repair an existing event from its clean raw attempts without model calls."""
    if not isinstance(memory, dict):
        return None
    repaired = dict(memory)
    raw = repaired.get("raw_response")
    raw = raw if isinstance(raw, dict) else {}
    attempts = raw.get("narrative_attempts") or []
    if raw.get("narrative"):
        attempts = [*attempts, raw["narrative"]]

    selected: tuple[str, dict[str, Any]] | None = None
    seen: set[str] = set()
    for attempt in attempts:
        if not isinstance(attempt, str) or attempt in seen:
            continue
        seen.add(attempt)
        fields = parse_narrative(attempt)
        if (
            semantic_text_clean(fields.get("where"), field="where")
            and semantic_text_clean(fields.get("what"), field="what")
            and (
                not fields.get("relations")
                or semantic_text_clean(fields.get("relations"), field="relations")
            )
        ):
            selected = (attempt, fields)
            break

    fallbacks = list(
        (repaired.get("semantic_validation") or {}).get("fallbacks") or []
    )
    if selected is not None:
        narrative, fields = selected
        repaired["where"] = fields["where"]
        repaired["what"] = fields["what"]
        repaired["narrative"] = narrative
        relationships = dict(repaired.get("dynamic_scene_relationships") or {})
        relationships["location_relationships"] = (
            [fields["relations"]] if fields.get("relations") else []
        )
        repaired["dynamic_scene_relationships"] = relationships
        if "narrative_from_clean_attempt" not in fallbacks:
            fallbacks.append("narrative_from_clean_attempt")
    else:
        clean_where = trim_semantic_contamination(repaired.get("where"))
        clean_what = trim_semantic_contamination(repaired.get("what"))
        if (
            semantic_text_clean(clean_where, field="where")
            and semantic_text_clean(clean_what, field="what")
        ):
            repaired["where"] = clean_where
            repaired["what"] = clean_what
            relationships = dict(repaired.get("dynamic_scene_relationships") or {})
            relations = relationships.get("location_relationships") or []
            clean_relations = [
                cleaned
                for relation in relations
                if (cleaned := trim_semantic_contamination(relation))
                and semantic_text_clean(cleaned, field="relations")
            ]
            relationships["location_relationships"] = clean_relations
            repaired["dynamic_scene_relationships"] = relationships
            repaired["narrative"] = (
                f"Where: {clean_where}; What: {clean_what}"
                + (
                    f"; Relations: {clean_relations[0]}"
                    if clean_relations
                    else ""
                )
            )
            if "trimmed_prompt_contamination" not in fallbacks:
                fallbacks.append("trimmed_prompt_contamination")

    if not semantic_text_clean(repaired.get("anomaly_entity"), field="entity"):
        repaired["anomaly_entity"] = entity_from_what(repaired.get("what"))
        repaired["anomaly_count"] = 1
        if "anomaly_from_what_count_one" not in fallbacks:
            fallbacks.append("anomaly_from_what_count_one")

    repaired["semantic_validation"] = {
        "complete": semantic_memory_complete(repaired),
        "fallbacks": fallbacks,
    }
    return repaired if semantic_memory_complete(repaired) else None
