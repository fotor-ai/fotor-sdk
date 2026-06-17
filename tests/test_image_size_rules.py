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

    def test_seedream_clamped(self) -> None:
        # seedream has no preferred_size mapping; we should clamp by max_long_side.
        w, h = _resolve_image_size(
            model_id="seedream-4-5-251128",
            aspect_ratio="1:1",
            resolution="1k",
        )
        self.assertEqual((w, h), (2048, 2048))
        self.assertLessEqual(max(w, h), 2048)

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

    def test_seedream_resolution_4k_clamped_long_side(self) -> None:
        w, h = _resolve_image_size(
            model_id="seedream-4-5-251128",
            aspect_ratio="16:9",
            resolution="4k",
        )
        self.assertLessEqual(max(w, h), 2048)
        self.assertEqual((w, h), (2048, 1152))

    def test_resolution_above_model_support_downgrades_to_highest_supported(self) -> None:
        w, h = _resolve_image_size(
            model_id="gpt-image-2",
            aspect_ratio="1:1",
            resolution="4k",
        )

        self.assertEqual((w, h), (2048, 2048))


if __name__ == "__main__":
    unittest.main()
