from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import tifffile

from droplet_impact_cv.analysis import analyze_sequence, prepare_sequence
from droplet_impact_cv.imaging import (
    foreground_segmentation, hysteresis_change_mask, make_structure,
    strong_drop_model, threshold_sampling_anchor,
)
from droplet_impact_cv.models import AnalysisConfig, SurfaceLine
from droplet_impact_cv.shadow import has_consistent_history


class ShadowSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((140, 160), 4000, dtype=np.float32)
        self.surface = SurfaceLine(110)
        self.config = AnalysisConfig(
            Path("input"), Path("out.csv"), strong_change_threshold=1000, weak_change_threshold=100,
            weak_growth_steps=20, surface_y_px=110, surface_angle_deg=0,
            top_exclusion_ratio=0, morphology_radius_px=1, foreground_depth_px=0,
            min_component_area_px2=20, min_gas_area_px2=5, min_impact_width_px=10,
            debug_dir=None, include_pre_impact=True,
        )
        self.structure = make_structure(1)

    def drop(self, center_y, shadow=False, sheet=False, bright=False):
        image = self.background.copy()
        cv2.ellipse(image, (80, center_y), (27, 30), 0, 0, 360, 3800, -1)
        cv2.ellipse(image, (80, center_y), (25, 28), 0, 0, 360, 2000, -1)
        if shadow:
            image[104:111, 52:112] = 3500
        if sheet:
            image[106:111, 40:126] = 4200 if bright else 3800
        return image

    def evidence(self, image, frame):
        return strong_drop_model(image, self.background, self.surface, 1000,
                                 self.config, self.structure, frame)[1]

    def segment(self, image, previous=(), config=None):
        return foreground_segmentation(
            image, self.background, self.surface, 1000, config or self.config,
            self.structure, 100, frame_number=40, previous_models=previous,
        )

    def history(self):
        return (self.evidence(self.drop(54), 20), self.evidence(self.drop(64), 30))

    def test_zero_reach_disables_extension_and_negative_reach_is_rejected(self):
        image = self.background.copy()
        image[30:40, 20:25] = 2000
        image[30:40, 25:75] = 3800
        mask = hysteresis_change_mask(image, self.background, 1000, 100, max_growth_steps=0)
        self.assertTrue(mask[35, 22])
        self.assertFalse(mask[35, 30])
        with self.assertRaises(ValueError):
            hysteresis_change_mask(image, self.background, 1000, 100, max_growth_steps=-1)

    def test_tracked_detached_drop_loses_shadow_contact_but_keeps_strong_pixels(self):
        image = self.drop(74, shadow=True)
        original = image.copy()
        result = self.segment(image, self.history())
        legacy = self.segment(image, config=replace(self.config, approach_shadow_guard=False))
        self.assertGreater(result.shadow_removed_px2, 0)
        self.assertFalse(result.mask[109:111].any())
        self.assertTrue(legacy.mask[109:111].any())
        self.assertTrue(result.mask[np.abs(image - self.background) >= 1000].all())
        self.assertIsNone(result.warning)
        np.testing.assert_array_equal(image, original)

    def test_missing_or_inconsistent_history_preserves_candidate_and_warns(self):
        image = self.drop(74, shadow=True)
        expected = self.segment(image, config=replace(self.config, approach_shadow_guard=False))
        wrong = tuple(replace(model, center=(model.center[0] + 40, model.center[1]))
                      for model in self.history())
        for previous in ((), self.history()[:1], wrong):
            result = self.segment(image, previous)
            np.testing.assert_array_equal(result.mask, expected.mask)
            self.assertEqual(result.shadow_removed_px2, 0)
            self.assertIn("Possible shadow contact", result.warning)

    def test_contacting_dark_and_bright_faint_sheets_are_preserved(self):
        config = replace(self.config, weak_growth_steps=40)
        for bright in (False, True):
            image = self.drop(88, sheet=True, bright=bright)
            result = self.segment(image, self.history(), config)
            expected = self.segment(image, config=replace(config, approach_shadow_guard=False))
            np.testing.assert_array_equal(result.mask, expected.mask)
            self.assertTrue(result.mask[108, 42])
            self.assertTrue(result.mask[108, 123])
            self.assertEqual(result.shadow_removed_px2, 0)

    def test_shadow_envelope_respects_tilted_surface(self):
        self.surface = SurfaceLine(110, 12)
        result = self.segment(self.drop(74, shadow=True), self.history())
        yy, xx = np.indices(result.mask.shape)
        contact_band = np.abs(yy - self.surface.y_at(xx, result.mask.shape[1])) <= 1
        self.assertGreater(result.shadow_removed_px2, 0)
        self.assertFalse(result.mask[contact_band].any())

    def test_initial_contact_rejects_diffuse_fringe_but_keeps_sharp_sheets(self):
        previous = (self.evidence(self.drop(63), 20), self.evidence(self.drop(73), 30))
        image = self.drop(83)
        yy, xx = np.indices(image.shape)
        shadow = 450 * np.exp(-((xx - 80) / 36) ** 2 / 2 - ((yy - 109) / 7) ** 2 / 2)
        diffuse = np.minimum(image, self.background - shadow).astype(np.float32)
        result = self.segment(diffuse, previous)
        self.assertIsNotNone(result.free_drop_model)
        self.assertFalse(result.free_drop_model.is_detached)
        self.assertGreater(result.shadow_removed_px2, 0)
        self.assertTrue(result.mask[109, 80])
        config = replace(self.config, weak_growth_steps=40)
        for value in (3800, 4200):
            sheet = image.copy()
            shelf = (yy >= 106) & (yy <= 110) & (xx >= 40) & (xx <= 125) & (sheet == self.background)
            sheet[shelf] = value
            guarded = self.segment(sheet, previous, config)
            self.assertIsNotNone(guarded.free_drop_model)
            self.assertFalse(guarded.free_drop_model.is_detached)
            ordinary = self.segment(sheet, config=replace(config, approach_shadow_guard=False))
            np.testing.assert_array_equal(guarded.mask, ordinary.mask)

    def test_history_uses_filename_intervals_and_rejects_size_changes(self):
        current = self.evidence(self.drop(74), 40)
        self.assertTrue(has_consistent_history(current, self.history()))
        self.assertFalse(has_consistent_history(replace(current, frame_number=80), self.history()))
        self.assertFalse(has_consistent_history(replace(current, axes=(45, 90)), self.history()))

    def test_clipped_and_deformed_drops_cannot_supply_free_drop_evidence(self):
        clipped = replace(self.config, top_exclusion_ratio=0.4)
        self.assertIsNone(strong_drop_model(self.drop(54), self.background, self.surface,
                                          1000, clipped, self.structure, 20)[1])
        image = self.background.copy()
        image[40:85, 40:120] = 2000
        self.assertIsNone(self.evidence(image, 20))

    def test_preview_seek_and_batch_share_shadow_rejection_and_contact_anchor(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            images = [self.background, self.drop(54), self.drop(64),
                      self.drop(74, shadow=True), self.drop(88, sheet=True)]
            files = []
            for number, image in zip((10, 20, 30, 40, 50), images):
                path = folder / f"frame{number:06d}.tif"
                tifffile.imwrite(path, image.astype(np.uint16))
                files.append(path)
            config = replace(self.config, input_dir=folder)
            prepared = prepare_sequence(config)
            # Seek directly to the shadow frame, before any sequential playback.
            _, sought = prepared.measure(files[3])
            sequential = prepare_sequence(config)
            for path in files[:4]:
                _, played = sequential.measure(path)
            np.testing.assert_array_equal(sought.mask, played.mask)
            self.assertGreater(sought.shadow_removed_px2, 0)
            self.assertEqual(sought.d_spread_px, 0)
            # A replaced preview shares the small model cache but must use its
            # own thresholds when preceding-frame evidence is requested.
            replace(prepared, weak_change_threshold=1000).measure(files[2])
            _, sought_again = replace(prepared, weak_change_threshold=100).measure(files[3])
            np.testing.assert_array_equal(sought_again.mask, sought.mask)
            rows = analyze_sequence(config)
            self.assertEqual(rows[3].shadow_removed_px2, sought.shadow_removed_px2)
            self.assertEqual({row.impact_frame for row in rows}, {50})
            self.assertIsNotNone(rows[4].segmentation_warning)
            anchor, method = threshold_sampling_anchor(
                files, self.background, self.surface, 1000, 100, config,
                self.structure, surface_is_calibrated=True,
            )
            self.assertEqual((anchor, method), (4, "surface-contact"))


if __name__ == "__main__":
    unittest.main()
