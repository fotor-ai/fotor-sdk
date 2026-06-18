import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from fotor_sdk.tasks import _resolve_image_size


class TestImageSizeRules(unittest.TestCase):
    def test_unknown_model_uses_conservative_fallback_sizes(self) -> None:
        w, h = _resolve_image_size(
            model_id="unknown-model",
            aspect_ratio="1:1",
            resolution="1k",
        )
        self.assertEqual((w, h), (1024, 1024))

        w, h = _resolve_image_size(
            model_id="unknown-model",
            aspect_ratio="16:9",
            resolution="1k",
        )
        self.assertEqual((w, h), (1392, 752))

    def test_unknown_model_nonstandard_ratio_uses_1k_pixel_budget(self) -> None:
        w, h = _resolve_image_size(
            model_id="unknown-model",
            aspect_ratio="7:5",
            resolution="1k",
        )
        self.assertEqual((w, h), (1213, 866))

    def test_seedream_4_5_uses_min_pixel_floor(self) -> None:
        from fotor_sdk import tasks

        model_id = "seedream-4-5-251128"
        original = tasks.MODEL_IMAGE_SIZE_RULES[model_id]
        tasks.MODEL_IMAGE_SIZE_RULES[model_id] = {
            **original,
            "resolution_default": "1k",
            "resolution_supports": ["1k"],
        }
        try:
            w, h = _resolve_image_size(
                model_id=model_id,
                aspect_ratio="16:9",
                resolution="1k",
            )
        finally:
            tasks.MODEL_IMAGE_SIZE_RULES[model_id] = original

        self.assertEqual((w, h), (2560, 1440))
        self.assertGreaterEqual(w * h, 3_686_400)

    def test_gemini_flash_image_default_1k_1_1(self) -> None:
        w, h = _resolve_image_size(
            model_id="gemini-2.5-flash-image",
            aspect_ratio="1:1",
            resolution="1k",
        )
        self.assertEqual((w, h), (1024, 1024))

    def test_gemini_3_pro_image_preview_default_1k_1_1(self) -> None:
        w, h = _resolve_image_size(
            model_id="gemini-3-pro-image-preview",
            aspect_ratio="1:1",
            resolution="1k",
        )
        self.assertEqual((w, h), (1024, 1024))

    def test_gpt_image_2_preferred_sizes(self) -> None:
        w, h = _resolve_image_size(
            model_id="gpt-image-2",
            aspect_ratio="1:1",
            resolution="1k",
        )
        self.assertEqual((w, h), (1024, 1024))

        w, h = _resolve_image_size(
            model_id="gpt-image-2",
            aspect_ratio="21:9",
            resolution="1k",
        )
        self.assertEqual((w, h), (1568, 672))

    def test_seedream_4_0_4k_allows_configured_max_pixels(self) -> None:
        w, h = _resolve_image_size(
            model_id="seedream-4-0-250828",
            aspect_ratio="1:1",
            resolution="4k",
        )
        self.assertEqual((w, h), (4096, 4096))
        self.assertLessEqual(w * h, 16_777_216)

    def test_seedream_4_5_uses_pixel_range_without_long_side_clamp(self) -> None:
        w, h = _resolve_image_size(
            model_id="seedream-4-5-251128",
            aspect_ratio="16:9",
            resolution="4k",
        )
        self.assertEqual((w, h), (5461, 3072))
        self.assertGreater(max(w, h), 4096)
        self.assertLessEqual(w * h, 16_777_216)

    def test_seedream_5_lite_uses_min_pixel_floor(self) -> None:
        from fotor_sdk import tasks

        model_id = "seedream-5-0-260128"
        original = tasks.MODEL_IMAGE_SIZE_RULES[model_id]
        tasks.MODEL_IMAGE_SIZE_RULES[model_id] = {
            **original,
            "resolution_default": "1k",
            "resolution_supports": ["1k"],
        }
        try:
            w, h = _resolve_image_size(
                model_id=model_id,
                aspect_ratio="16:9",
                resolution="1k",
            )
        finally:
            tasks.MODEL_IMAGE_SIZE_RULES[model_id] = original

        self.assertEqual((w, h), (2560, 1440))
        self.assertGreaterEqual(w * h, 3_686_400)

    def test_resolution_above_model_support_downgrades_to_highest_supported(self) -> None:
        w, h = _resolve_image_size(
            model_id="gpt-image-2",
            aspect_ratio="1:1",
            resolution="4k",
        )

        self.assertEqual((w, h), (2048, 2048))


if __name__ == "__main__":
    unittest.main()
