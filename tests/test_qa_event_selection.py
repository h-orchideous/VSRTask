from __future__ import annotations

import unittest

from vsrtask.qa_event_selection import (
    event_matches_entity,
    extract_query_entity,
    select_events,
)


def event(
    frame: int,
    score: float,
    what: str,
    objects: list[str] | None = None,
) -> dict:
    return {
        "peak_frame": frame,
        "peak_score": score,
        "episodic_memory": {
            "what": what,
            "objects": [
                {"name": name, "count": 1} for name in (objects or [])
            ],
        },
    }


class QaEventSelectionTest(unittest.TestCase):
    def test_extracts_vsr_query_entity(self) -> None:
        question = (
            "These are frames of a video. Which option correctly represents the "
            "order in which the bouquet of red roses appeared in the video?"
        )
        self.assertEqual(extract_query_entity(question), "bouquet of red roses")

    def test_entity_alias_matches_semantic_object(self) -> None:
        self.assertTrue(
            event_matches_entity(
                event(3, 1.0, "A dog standing on a chair", ["dog", "chair"]),
                "Golden Retriever",
            )
        )

    def test_entity_match_reads_structured_anomaly(self) -> None:
        structured = event(3, 1.0, "unknown")
        structured["episodic_memory"] = {
            "semantic_schema_version": 3,
            "anomaly": {
                "entity": "golden retriever",
                "description": "A golden retriever figurine",
                "count": 1,
            },
        }
        self.assertTrue(event_matches_entity(structured, "Golden Retriever"))

    def test_threshold_and_entity_retrieval_preserve_time_order(self) -> None:
        memory = {
            "events": [
                event(30, 0.95, "Pikachu on a bed"),
                event(10, 0.89, "Pikachu in a bin"),
                event(20, 0.97, "Charlie Chaplin poster"),
                event(15, 0.92, "Pikachu on a chair"),
            ]
        }
        selection = select_events(
            memory,
            "Which order in which the Pikachu appeared in the video?",
            score_threshold=0.90,
            retrieval="entity",
        )

        self.assertEqual([item["peak_frame"] for item in selection.events], [15, 30])
        self.assertEqual(selection.source_event_count, 4)
        self.assertEqual(selection.threshold_event_count, 3)
        self.assertEqual(selection.retrieved_event_count, 2)

    def test_no_entity_match_falls_back_to_threshold_events(self) -> None:
        memory = {"events": [event(20, 0.95, "A poster on a wall")]}
        selection = select_events(
            memory,
            "Which order in which the Stitch appeared in the video?",
            score_threshold=0.90,
            retrieval="entity",
        )

        self.assertEqual(len(selection.events), 1)
        self.assertEqual(
            selection.retrieval_fallback, "no_entity_match_use_threshold_events"
        )


if __name__ == "__main__":
    unittest.main()
