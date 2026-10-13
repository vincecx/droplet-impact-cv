from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from droplet_impact_cv.analysis import prepare_sequence
from droplet_impact_cv.imaging import estimate_threshold, make_structure, threshold_sampling_anchor
from droplet_impact_cv.models import AnalysisConfig, SurfaceLine


class AutoCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((120, 120), 4000, dtype=np.float32)
        self.config = AnalysisConfig(
            Path("synthetic"), Path("out.csv"), top_exclusion_ratio=0,
            min_droplet_area_px2=20, min_contact_px=5, min_impact_width_px=10,
            morphology_radius_px=1,
        )

    def ellipse(self, center_y=45, deformed=False):
        mask = np.zeros(self.background.shape, dtype=np.uint8)
        cv2.ellipse(mask, (60, center_y), (20, 25), 0, 0, 360, 1, -1)
        if deformed:
            mask[center_y + 1:] = 0
        return np.where(mask, 2000, self.background).astype(np.float32)

    def files_and_reader(self, images):
        # Gaps in filename numbers must not be confused with sample positions.
        files = [Path(f"capture_{10 * (index + 1):06d}.tif") for index in range(len(images))]
        images_by_file = dict(zip(files, images))
        return files, images_by_file.__getitem__

    def anchor(self, images, calibrated=False, config=None):
        files, reader = self.files_and_reader(images)
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader):
            return threshold_sampling_anchor(
                files, self.background, SurfaceLine(70), 250, 75,
                config or self.config, make_structure(1), calibrated,
            )

    def test_shape_change_needs_stable_baseline_and_consecutive_confirmation(self):
        round_drop = self.ellipse()
        deformed = self.ellipse(deformed=True)
        self.assertEqual(self.anchor([round_drop] * 3 + [deformed] * 2), (3, "shape-change"))
        self.assertEqual(self.anchor([round_drop] * 3 + [deformed, round_drop]), (None, "range-sampling"))
        self.assertEqual(self.anchor([deformed] * 5), (None, "range-sampling"))

    def test_clipped_drop_is_not_a_stable_ellipse_baseline(self):
        config = replace(self.config, top_exclusion_ratio=0.3)
        self.assertEqual(
            self.anchor([self.ellipse()] * 3 + [self.ellipse(deformed=True)] * 2, config=config),
            (None, "range-sampling"),
        )

    def test_calibrated_contact_does_not_require_shape_deformation(self):
        self.assertEqual(
            self.anchor([self.ellipse(30), self.ellipse(40), self.ellipse(55)], calibrated=True),
            (2, "surface-contact"),
        )

    def test_late_noise_changes_receive_equal_frame_weight(self):
        # No MAD in these uniform fields: each threshold is its frame's delta.
        images = [self.background - delta for delta in (0, 10, 40, 400)]
        files, reader = self.files_and_reader(images)
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader):
            threshold = estimate_threshold(files, self.background, 100, 4, 200)
        self.assertAlmostEqual(threshold, 292)

    def test_empty_noise_roi_respects_floor(self):
        files, _ = self.files_and_reader([self.background])
        self.assertEqual(estimate_threshold(files, self.background, 10, 1, 200), 200)

    def test_tilted_surface_excludes_substrate_noise(self):
        yy, xx = np.indices(self.background.shape)
        line = SurfaceLine(60, 20)
        image = self.background.copy()
        image[yy >= line.y_at(xx, image.shape[1]) - 30] -= 1800
        files, reader = self.files_and_reader([image])
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader):
            threshold = estimate_threshold(files, self.background, 60, 1, 200, surface_line=line)
        self.assertEqual(threshold, 200)

    def test_anchor_window_is_clipped_and_uses_actual_file_numbers(self):
        images = [self.background, self.ellipse(30), self.ellipse(40), self.ellipse(55)]
        files, reader = self.files_and_reader(images)
        config = replace(self.config, surface_y_px=70, surface_angle_deg=0)
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader):
            prepared = prepare_sequence(config, files, self.background)
        self.assertEqual(prepared.threshold_anchor_frame, 40)
        self.assertEqual(prepared.threshold_sample_frames, (20, 30, 40))

    def test_anchor_window_has_two_before_anchor_and_seven_after(self):
        files, reader = self.files_and_reader([self.background] * 20)
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader), patch(
            "droplet_impact_cv.analysis.threshold_sampling_anchor", return_value=(5, "shape-change")
        ):
            prepared = prepare_sequence(self.config, files, self.background)
        self.assertEqual(prepared.threshold_anchor_frame, 60)
        self.assertEqual(prepared.threshold_sample_frames, tuple(range(40, 131, 10)))

    def test_no_anchor_samples_entire_range_and_manual_threshold_bypasses_scan(self):
        files, reader = self.files_and_reader([self.background] * 20)
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader):
            prepared = prepare_sequence(self.config, files, self.background)
        self.assertEqual(prepared.threshold_sampling_method, "range-sampling")
        self.assertEqual(len(prepared.threshold_sample_frames), 8)
        self.assertEqual(prepared.threshold_sample_frames[0], 10)
        self.assertEqual(prepared.threshold_sample_frames[-1], 200)
        self.assertFalse(prepared.surface_is_calibrated)
        with patch("droplet_impact_cv.analysis.threshold_sampling_anchor") as scan:
            manual = prepare_sequence(replace(self.config, strong_threshold=450), files, self.background)
        scan.assert_not_called()
        self.assertEqual(manual.threshold, 450)
        self.assertEqual(manual.threshold_sample_frames, ())

    def test_complete_manual_surface_bypasses_silhouette_fitting(self):
        files, reader = self.files_and_reader([self.background, self.ellipse()])
        config = replace(
            self.config, surface_calibration_frame=20, surface_y_px=70,
            surface_angle_deg=-1, reflection_mode="none", strong_threshold=450,
        )
        with patch("droplet_impact_cv.imaging.read_image", side_effect=reader), patch(
            "droplet_impact_cv.analysis.estimate_surface_line_from_nonreflection_frame",
            side_effect=ValueError("Unusable silhouette"),
        ) as fit:
            prepared = prepare_sequence(config, files, self.background)
        fit.assert_not_called()
        self.assertEqual(prepared.surface_line, SurfaceLine(70, -1))


if __name__ == "__main__":
    unittest.main()
