from __future__ import annotations

import unittest

from PIL import Image

from scripts.caption_event_memory_qwen25 import parse_counts
from vsrtask.semantic_evidence import build_detector_focus_crop
from vsrtask.semantic_memory import (
    parse_physical_where_response,
    recover_unstructured_observation,
    semantic_memory_complete,
)


class SemanticEvidenceTest(unittest.TestCase):
    def test_zero_count_objects_are_ignored(self) -> None:
        self.assertEqual(
            parse_counts("cat=1; baseball cap=0; chair=2"),
            [
                {"name": "cat", "count": 1},
                {"name": "chair", "count": 2},
            ],
        )

    def test_single_character_count_noise_is_ignored(self) -> None:
        self.assertEqual(
            parse_counts("panda bear=1; p=1; tv=1"),
            [
                {"name": "panda bear", "count": 1},
                {"name": "tv", "count": 1},
            ],
        )

    def test_focus_crop_uses_clean_source_pixels(self) -> None:
        image = Image.new("RGB", (230, 170), (12, 34, 56))
        crop = build_detector_focus_crop(image, [12], [17, 23])

        self.assertIsNotNone(crop)
        self.assertEqual(crop.getpixel((0, 0)), (12, 34, 56))
        self.assertLess(crop.width, image.width)
        self.assertLess(crop.height, image.height)

    def test_overlay_artifact_is_not_valid_semantic_memory(self) -> None:
        memory = {
            "semantic_schema_version": 2,
            "what": "Red square markers",
            "where": "around a poster",
            "scene": "room",
            "background": "wooden wall",
            "objects": [{"name": "square marker", "count": 20}],
        }

        self.assertFalse(semantic_memory_complete(memory))

    def test_overlay_colored_patch_is_not_valid_semantic_memory(self) -> None:
        memory = {
            "semantic_schema_version": 2,
            "what": "red squares around a poster",
            "where": "on a wall",
            "scene": "room",
            "background": "wooden panel",
            "objects": [{"name": "red square", "count": 20}],
        }

        self.assertFalse(semantic_memory_complete(memory))

    def test_template_placeholders_are_not_valid_semantic_memory(self) -> None:
        memory = {
            "semantic_schema_version": 2,
            "what": "<a yellow cartoon character>",
            "where": "inside a bin",
            "scene": "room",
            "background": "sofa",
            "objects": [{"name": "cartoon character", "count": 1}],
        }

        self.assertFalse(semantic_memory_complete(memory))

    def test_physical_objects_remain_valid(self) -> None:
        memory = {
            "semantic_schema_version": 2,
            "what": "a yellow plush toy",
            "where": "inside a bin",
            "scene": "living room",
            "background": "sofa and side table",
            "objects": [{"name": "plush toy", "count": 1}],
        }

        self.assertTrue(semantic_memory_complete(memory))

    def test_unstructured_observation_is_recovered(self) -> None:
        recovered = recover_unstructured_observation(
            "Stitch plush toy, bed, bedroom, wall art depicting trees"
        )

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["what"], "Stitch plush toy")
        self.assertEqual(recovered["scene"].lower(), "bedroom")
        self.assertTrue(semantic_memory_complete({
            "semantic_schema_version": 2,
            **recovered,
            "objects": [{"name": "stitch plush toy", "count": 1}],
        }))

    def test_contaminated_unstructured_observation_is_not_recovered(self) -> None:
        self.assertIsNone(
            recover_unstructured_observation(
                "red square markers around a poster in a room"
            )
        )

    def test_physical_where_parser_keeps_anchor_relation(self) -> None:
        self.assertEqual(
            parse_physical_where_response("Where: inside the trash can."),
            "inside the trash can",
        )

    def test_physical_where_parser_does_not_truncate_ladder(self) -> None:
        self.assertEqual(
            parse_physical_where_response("Where: on the ladder."),
            "on the ladder",
        )

    def test_physical_where_parser_rejects_frame_coordinates(self) -> None:
        self.assertIsNone(parse_physical_where_response("center of the frame"))


if __name__ == "__main__":
    unittest.main()
