from __future__ import annotations

import unittest

from vsrtask.semantic_memory import (
    anomaly_description_clean,
    build_structured_event_memory,
    choose_anomaly_description,
    choose_placement_text,
    merge_observation_attempts,
    parse_anomaly_response,
    parse_full_frame_observation,
    parse_grounded_placement,
    semantic_memory_complete,
)


class SemanticMemoryTest(unittest.TestCase):
    def test_normalizes_dedicated_anomaly_response(self) -> None:
        self.assertEqual(
            parse_anomaly_response("Foreground Object: A: Pikachu plush toy."),
            "Pikachu plush toy",
        )
        self.assertEqual(
            parse_anomaly_response(
                "Pikachu plush toy - on sofa armrest - living room"
            ),
            "Pikachu plush toy",
        )
        self.assertEqual(
            parse_anomaly_response("pl\n addCriterion\n plastic bag"),
            "plastic bag",
        )
        self.assertIsNone(parse_anomaly_response("pl\n addCriterion"))

    def test_prefers_specific_observation_over_generic_anomaly(self) -> None:
        self.assertEqual(
            choose_anomaly_description(
                "Stuffed toy",
                "Stuffed toy resembling Stitch from Lilo and Stitch",
            ),
            "Stuffed toy resembling Stitch from Lilo and Stitch",
        )
        self.assertEqual(
            choose_anomaly_description("Stitch figurine", "lamp"),
            "Stitch figurine",
        )

    def test_placement_uses_anchor_agreement_before_targeted_answer(self) -> None:
        self.assertEqual(
            choose_placement_text(
                "nightstand",
                "Stitch figurine on nightstand",
                "on the table",
            ),
            "on nightstand",
        )
        self.assertEqual(
            choose_placement_text("unknown", "Pikachu", "inside the trash can"),
            "inside the trash can",
        )

    def test_rejects_option_letter_and_template_anomaly_text(self) -> None:
        self.assertFalse(anomaly_description_clean("C"))
        self.assertFalse(
            anomaly_description_clean(
                "A concise anomaly, visible attributes: yellow plush toy"
            )
        )
        self.assertFalse(anomaly_description_clean("Foreground Object: A"))
        self.assertFalse(
            anomaly_description_clean("wood Foreground: wooden floor")
        )
        self.assertFalse(
            anomaly_description_clean("Pikachu toy Where on: kitchen counter")
        )
        self.assertFalse(anomaly_description_clean("stitch||bedroom||nighttime"))
        self.assertTrue(anomaly_description_clean("yellow Pikachu plush toy"))

    def test_rejects_control_tokens_in_context(self) -> None:
        fields = {
            "what": "Stitch plush toy",
            "where": "on a white bedspread",
            "scene": "Bedroom setting\nmatchCondition",
            "background": "unknown",
        }
        objects = [{"name": "stitch", "count": 1}]
        memory = {
            "semantic_schema_version": 3,
            **fields,
            "objects": objects,
            **build_structured_event_memory(fields, objects),
        }
        self.assertFalse(semantic_memory_complete(memory))

    def test_parses_image_conditioned_pipe_retry(self) -> None:
        self.assertEqual(
            parse_full_frame_observation(
                "Stitch plush toy || on the bed || bedroom || white wall and lamp"
            ),
            {
                "what": "Stitch plush toy",
                "where": "on the bed",
                "scene": "bedroom",
                "background": "white wall and lamp",
            },
        )

    def test_parses_qwen_label_variants(self) -> None:
        self.assertEqual(
            parse_full_frame_observation(
                "What: A Stitch plush toy on bed. "
                "Where on: top of neatly made white bed sheets. "
                "Scene in: bedroom setting. "
                "Background includes: wooden headboard and pillows."
            ),
            {
                "what": "A Stitch plush toy on bed",
                "where": "top of neatly made white bed sheets",
                "scene": "bedroom setting",
                "background": "wooden headboard and pillows",
            },
        )

    def test_selects_clean_candidate_after_contaminated_duplicate(self) -> None:
        parsed = parse_full_frame_observation(
            "Foreground Object: A addCriterion "
            "Foreground Object: Stuffed toy resembling Stitch addCriterion "
            "Where: on a white bedspread. Scene: bedroom. "
            "Background: white pillows and wooden headboard."
        )
        self.assertEqual(parsed["what"], "Stuffed toy resembling Stitch")
        self.assertEqual(parsed["where"], "on a white bedspread")

    def test_merges_valid_primary_fields_with_unknown_context(self) -> None:
        merged = merge_observation_attempts(
            [
                "Foreground Object: A addCriterion "
                "Foreground Object: Stuffed toy resembling Stitch addCriterion "
                "Where: on a white bedspread. Scene: bedroom.",
                "What: A addCriterion",
                "stitch||bedroom||nighttime",
            ]
        )
        self.assertIsNotNone(merged)
        fields, missing = merged
        self.assertEqual(fields["what"], "Stuffed toy resembling Stitch")
        self.assertEqual(fields["where"], "on a white bedspread")
        self.assertEqual(fields["background"], "unknown")
        self.assertEqual(missing, ["background"])

    def test_parses_relation_and_anchor(self) -> None:
        self.assertEqual(
            parse_grounded_placement("On the wooden chair."),
            {
                "relation": "on",
                "anchor": "the wooden chair",
                "text": "on the wooden chair",
            },
        )

    def test_rejects_frame_coordinates_and_vague_scene(self) -> None:
        self.assertIsNone(parse_grounded_placement("center of the frame"))
        self.assertIsNone(parse_grounded_placement("visible in the depicted scene"))
        self.assertIsNone(parse_grounded_placement("office setting"))

    def test_targeted_retry_replaces_unresolved_observation(self) -> None:
        structured = build_structured_event_memory(
            {
                "what": "Pikachu toy",
                "where": "center of the frame",
                "scene": "bedroom",
                "background": "A bed and a nightstand",
            },
            [{"name": "pikachu", "count": 1}, {"name": "bed", "count": 1}],
            targeted_where="on the bed",
        )
        self.assertEqual(structured["placement"]["status"], "targeted_retry")
        self.assertEqual(structured["placement"]["anchor"], "the bed")
        self.assertEqual(structured["anomaly"]["entity"], "pikachu toy")
        self.assertEqual(structured["context"]["objects"], [{"name": "bed", "count": 1}])

    def test_v3_memory_validation(self) -> None:
        fields = {
            "what": "Pikachu toy",
            "where": "on the bed",
            "scene": "bedroom",
            "background": "A bed and a nightstand",
        }
        objects = [{"name": "pikachu", "count": 1}, {"name": "bed", "count": 1}]
        memory = {
            "semantic_schema_version": 3,
            **fields,
            "objects": objects,
            **build_structured_event_memory(fields, objects),
        }
        self.assertTrue(semantic_memory_complete(memory))


if __name__ == "__main__":
    unittest.main()
