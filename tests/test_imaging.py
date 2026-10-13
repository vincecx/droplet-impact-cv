from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import tifffile

from droplet_impact_cv.imaging import (
    build_background,
    component_measurement,
    estimate_contact_line,
    estimate_surface_line_from_nonreflection_frame,
    select_liquid_component,
    estimate_vertical_symmetry_y,
    foreground_mask,
    find_image_files,
    frame_number_from_filename,
    make_structure,
    read_image,
    select_calibration_component,
    select_symmetric_calibration_component,
    touches_surface,
)
from droplet_impact_cv.models import (
    DEFAULT_MIN_STRONG_THRESHOLD,
    AnalysisConfig,
    SurfaceLine,
)


class BackgroundTests(unittest.TestCase):
    def test_frame_number_uses_last_six_digits_after_numeric_experiment_name(self):
        self.assertEqual(frame_number_from_filename(Path("2026.9.14 Al-60-5 最低高度 常温3000010.tif")), 10)
        self.assertEqual(frame_number_from_filename(Path("capture_000005.jpg")), 5)
        with self.assertRaises(ValueError):
            frame_number_from_filename(Path("capture_00005.tif"))

    def test_mirror_candidate_depth_uses_configured_cutoff_inclusively(self):
        mask = np.zeros((120, 100), dtype=bool)
        mask[20:76, 30:70] = True
        config = AnalysisConfig(input_dir=Path("input"), output_csv=Path("out.csv"))
        self.assertIsNone(select_calibration_component(mask, 50, config))
        selected = select_calibration_component(mask, 50, replace(config, mirror_min_depth_px=25))
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[0], mask)
        self.assertIsNone(select_calibration_component(mask, 50, replace(config, mirror_min_depth_px=26)))

    def test_background_uses_only_the_smallest_numbered_frame(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            first = input_dir / "capture_000001.tif"
            second = input_dir / "capture_000002.tif"
            tifffile.imwrite(first, np.full((4, 5), 1000, dtype=np.uint16))
            tifffile.imwrite(second, np.full((4, 5), 3000, dtype=np.uint16))

            background = build_background([second, first])

        np.testing.assert_array_equal(
            background,
            np.full((4, 5), 1000, dtype=np.float32),
        )

    def test_jpeg_frames_are_discovered_and_scaled_to_12_bit(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            jpeg_path = input_dir / "capture_000001.jpg"
            cv2.imwrite(str(jpeg_path), np.full((8, 9), 255, dtype=np.uint8))

            files = find_image_files(input_dir)
            image = read_image(files[0])

        self.assertEqual(files, [jpeg_path])
        self.assertEqual(image.shape, (8, 9))
        self.assertAlmostEqual(float(np.median(image)), 4095.0, delta=20.0)


class ForegroundMaskTests(unittest.TestCase):
    def test_top_exclusion_clips_strong_weak_and_cleaned_pixels(self) -> None:
        background = np.full((41, 40), 4000, dtype=np.float32)
        image = background.copy()
        image[2:8, 5:15] = 2000  # An excluded seed must not grow into weak pixels.
        image[8:20, 5:15] = 3400
        image[2:30, 22:35] = 2000  # A connected contour crosses the cutoff.
        config = AnalysisConfig(input_dir=Path("input"), output_csv=Path("out.csv"))
        original = background.copy()
        for ratio, cutoff in ((0, 0), (0.25, 11), (0.5, 21), (1, 41)):
            with self.subTest(ratio=ratio):
                mask = foreground_mask(
                    image, background, SurfaceLine(35, 0), 1500,
                    replace(config, top_exclusion_ratio=ratio), make_structure(2),
                    weak_threshold=500,
                )
                self.assertFalse(mask[:cutoff].any())
                if ratio < 1:
                    self.assertTrue(mask[25, 28])
                if ratio == 0:
                    self.assertTrue(mask[15, 10])
                else:
                    self.assertFalse(mask[15, 10])
                measurement = component_measurement(
                    mask, SurfaceLine(35, 0), replace(config, top_exclusion_ratio=ratio, min_droplet_area_px2=1)
                )
                self.assertFalse(measurement.mask[:cutoff].any())
        np.testing.assert_array_equal(background, original)

    def test_excluded_component_cannot_win_selection_or_fill_back_in(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:8, :] = 1
        mask[15:25, 15:25] = 1
        config = AnalysisConfig(input_dir=Path("input"), output_csv=Path("out.csv"), min_droplet_area_px2=1)
        measurement = component_measurement(mask, SurfaceLine(30, 0), config)
        self.assertEqual(measurement.area_px2, 100)
        self.assertFalse(measurement.mask[:10].any())
        self.assertTrue(mask[:8].all())  # The caller's mask is preserved.
        empty = component_measurement(mask, SurfaceLine(30, 0), replace(config, top_exclusion_ratio=1))
        self.assertEqual(empty.area_px2, 0)
        self.assertIsNone(empty.d_eq_volume_px)

    def test_default_threshold_excludes_weak_shadow_attached_to_droplet(self) -> None:
        background = np.full((60, 60), 4000, dtype=np.float32)
        image = background.copy()
        image[15:40, 20:40] = 2000  # Droplet: delta 2000.
        image[40:43, 20:40] = 3200  # Attached weak shadow: delta 800.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            debug_dir=None,
        )

        mask = foreground_mask(
            image,
            background,
            SurfaceLine(45.0, angle_deg=0.0),
            DEFAULT_MIN_STRONG_THRESHOLD,
            config,
            make_structure(1),
        )

        self.assertEqual(mask[30, 30], 1)
        self.assertEqual(mask[41, 30], 0)

    def test_bright_droplet_is_detected_as_foreground(self) -> None:
        background = np.full((60, 60), 1000, dtype=np.float32)
        image = background.copy()
        image[15:40, 20:40] = 3000  # Refracted backlight: brighter than background.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            debug_dir=None,
        )

        mask = foreground_mask(
            image,
            background,
            SurfaceLine(45.0, angle_deg=0.0),
            DEFAULT_MIN_STRONG_THRESHOLD,
            config,
            make_structure(1),
        )

        self.assertEqual(mask[30, 30], 1)

    def test_weak_threshold_extends_strong_detections_only_when_connected(self) -> None:
        background = np.full((60, 90), 4000, dtype=np.float32)
        image = background.copy()
        image[20:40, 10:20] = 2000  # Strong rim: delta 2000.
        image[20:40, 20:40] = 3400  # Faint film interior attached to the rim: delta 600.
        image[20:40, 60:80] = 3400  # Isolated faint artifact: delta 600.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            debug_dir=None,
        )

        mask = foreground_mask(
            image,
            background,
            SurfaceLine(45.0, angle_deg=0.0),
            DEFAULT_MIN_STRONG_THRESHOLD,
            config,
            make_structure(1),
            weak_threshold=500.0,
        )

        self.assertEqual(mask[30, 15], 1)
        self.assertEqual(mask[30, 30], 1)
        self.assertEqual(mask[30, 70], 0)

    def test_weak_extension_growth_is_distance_bounded(self) -> None:
        background = np.full((30, 200), 4000, dtype=np.float32)
        image = background.copy()
        image[10:20, 5:10] = 2000  # Strong seed.
        image[10:20, 10:190] = 3400  # Long faint strip attached to the seed.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            debug_dir=None,
        )

        mask = foreground_mask(
            image,
            background,
            SurfaceLine(25.0, angle_deg=0.0),
            DEFAULT_MIN_STRONG_THRESHOLD,
            config,
            make_structure(1),
            weak_threshold=500.0,
        )

        self.assertEqual(mask[15, 30], 1)
        self.assertEqual(mask[15, 180], 0)

    def test_connected_thin_contact_boundary_survives_cleanup(self) -> None:
        background = np.full((80, 100), 4000, dtype=np.float32)
        dark_foreground = np.zeros(background.shape, dtype=np.uint8)
        dark_foreground[20:51, 40:81] = 1
        cv2.line(dark_foreground, (40, 20), (20, 50), 1, thickness=2)
        cv2.line(dark_foreground, (5, 10), (5, 40), 1, thickness=2)
        image = background.copy()
        image[dark_foreground > 0] = 2000
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )
        surface_line = SurfaceLine(50.0, angle_deg=0.0)

        mask = foreground_mask(
            image,
            background,
            surface_line,
            DEFAULT_MIN_STRONG_THRESHOLD,
            config,
            make_structure(3),
        )
        measurement = component_measurement(mask, surface_line, config)

        self.assertEqual(mask[50, 20], 1)
        self.assertEqual(mask[20, 5], 0)
        self.assertEqual(measurement.d_spread_px, 62.0)


class ComponentMeasurementTests(unittest.TestCase):
    def test_calibrated_center_selects_target_over_closer_drop(self) -> None:
        mask = np.zeros((100, 160), dtype=np.uint8)
        # The left droplet is closer to the surface, but calibration identifies
        # the target as the central droplet.
        mask[45:51, 5:35] = 1
        mask[20:40, 90:120] = 1
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        component, _area = select_liquid_component(
            mask, SurfaceLine(50.0, 0.0), config, preferred_center_x=105.0
        )

        self.assertTrue(component[30, 105])
        self.assertFalse(component[48, 20])

    def test_impact_detection_always_uses_contact_line_diameter(self) -> None:
        mask = np.zeros((100, 140), dtype=np.uint8)
        # The lifted rim is deliberately wider than the contact line.
        mask[30:45, 30:110] = 1
        mask[45:51, 50:90] = 1
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            min_impact_width_px=60,
            rim_to_rim=True,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, 0.0), config)

        self.assertEqual(measurement.d_contact_px, 40.0)
        self.assertEqual(measurement.d_rim_px, 80.0)
        self.assertFalse(touches_surface(measurement, config))

    def test_rim_to_rim_measurement_uses_rim_to_rim_width(self) -> None:
        mask = np.zeros((100, 140), dtype=np.uint8)
        # The footprint at the surface is 40 px, while the lifted rim span is
        # 80 px. A rim_to_rim measurement must use the latter.
        mask[30:45, 30:110] = 1
        mask[45:51, 50:90] = 1
        surface_line = SurfaceLine(50.0, angle_deg=0.0)
        contact_config = AnalysisConfig(
            input_dir=Path("input"), output_csv=Path("output.csv"),
            min_droplet_area_px2=1, min_contact_px=1, debug_dir=None,
        )
        rim_to_rim_config = AnalysisConfig(
            input_dir=Path("input"), output_csv=Path("output.csv"),
            min_droplet_area_px2=1, min_contact_px=1, rim_to_rim=True, debug_dir=None,
        )

        contact_measurement = component_measurement(mask, surface_line, contact_config)
        rim_measurement = component_measurement(mask, surface_line, rim_to_rim_config)

        self.assertEqual(contact_measurement.d_spread_px, 40.0)
        self.assertEqual(rim_measurement.d_spread_px, 80.0)
        self.assertEqual(rim_measurement.d_contact_px, 40.0)
        self.assertEqual(rim_measurement.d_rim_px, 80.0)
        self.assertEqual(rim_measurement.contact_bbox, (50, 50, 89, 50))
        self.assertEqual(rim_measurement.rim_bbox, (30, 30, 109, 30))

    def test_rim_to_rim_rim_measurement_is_parallel_to_surface(self) -> None:
        mask = np.zeros((160, 180), dtype=np.uint8)
        # A 0.5-slope, 80-px-wide parallelogram: the longest diameter must
        # follow the surface direction, rather than the image x axis.
        cv2.fillConvexPoly(
            mask,
            np.array([[30, 40], [110, 80], [100, 100], [20, 60]], dtype=np.int32),
            1,
        )
        surface_line = SurfaceLine(80.0, angle_deg=np.degrees(np.arctan(0.5)))
        config = AnalysisConfig(
            input_dir=Path("input"), output_csv=Path("output.csv"),
            min_droplet_area_px2=1, min_contact_px=1, rim_to_rim=True, debug_dir=None,
        )

        measurement = component_measurement(mask, surface_line, config)

        assert measurement.rim_bbox is not None
        x0, y0, x1, y1 = measurement.rim_bbox
        self.assertAlmostEqual(y1 - y0, 0.5 * (x1 - x0), delta=1.0)
        self.assertGreater(measurement.d_spread_px, 88.0)

    def test_area_equivalent_diameter_uses_filled_contour_in_both_modes(self) -> None:
        mask = np.zeros((120, 140), dtype=np.uint8)
        mask[20:50, 30:70] = 1
        mask[25:45, 40:60] = 0  # Enclosed bright interior belongs to the drop.
        for rim_to_rim in (False, True):
            with self.subTest(rim_to_rim=rim_to_rim):
                config = AnalysisConfig(
                    input_dir=Path("input"), output_csv=Path("output.csv"),
                    min_droplet_area_px2=1, min_contact_px=1,
                    rim_to_rim=rim_to_rim, debug_dir=None, top_exclusion_ratio=0,
                )
                measurement = component_measurement(
                    mask, SurfaceLine(70.0, angle_deg=0.0), config
                )
                self.assertEqual(measurement.area_px2, 1200)
                self.assertAlmostEqual(measurement.d_eq_area_px, np.sqrt(4 * 1200 / np.pi))
                self.assertNotAlmostEqual(measurement.d_eq_area_px, measurement.d_eq_volume_px)

    def test_empty_contour_has_no_equivalent_diameters(self) -> None:
        config = AnalysisConfig(input_dir=Path("input"), output_csv=Path("output.csv"))
        measurement = component_measurement(
            np.zeros((80, 100), dtype=bool), SurfaceLine(60.0), config
        )
        self.assertIsNone(measurement.d_eq_volume_px)
        self.assertIsNone(measurement.d_eq_area_px)

    def test_surface_aligned_dimensions_define_equivalent_diameter_and_clearance(self) -> None:
        mask = np.zeros((120, 140), dtype=np.uint8)
        # 40 px parallel to, 30 px normal to, and 21 px above the surface.
        mask[20:50, 30:70] = 1
        config = AnalysisConfig(
            input_dir=Path("input"), output_csv=Path("output.csv"),
            min_droplet_area_px2=1, min_contact_px=1, debug_dir=None, top_exclusion_ratio=0,
        )

        measurement = component_measurement(mask, SurfaceLine(70.0, angle_deg=0.0), config)

        self.assertEqual(measurement.d_parallel_px, 40.0)
        self.assertEqual(measurement.d_normal_px, 30.0)
        self.assertAlmostEqual(measurement.d_eq_volume_px or 0.0, 36.3424, places=4)
        self.assertEqual(measurement.surface_clearance_px, 21.0)
        self.assertEqual(measurement.surface_parallel_line, (30, 34, 69, 34))
        self.assertEqual(measurement.surface_normal_line, (50, 20, 50, 49))

    def test_overlay_is_clipped_at_surface_without_changing_contact_width(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:60, 20:80] = 1
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )
        surface_line = SurfaceLine(50.0, angle_deg=0.0)

        measurement = component_measurement(mask, surface_line, config)

        yy, xx = np.indices(measurement.mask.shape)
        distance_from_surface = yy - surface_line.y_at(xx, measurement.mask.shape[1])
        self.assertFalse(np.any(measurement.mask & (distance_from_surface > 0)))
        self.assertEqual(measurement.d_spread_px, 60.0)

    def test_bright_gap_open_at_surface_is_filled_as_droplet_interior(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:55, 20:80] = 1
        mask[45:55, 45:55] = 0
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, angle_deg=0.0), config)

        self.assertEqual(measurement.d_spread_px, 60.0)
        self.assertTrue(measurement.mask[50, 50])

    def test_component_clear_of_surface_band_measures_zero(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:44, 20:80] = 1  # Bottom row 43: 7 px above the line, outside +-3.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            reflection_mode="none",
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, 0.0), config)

        self.assertEqual(measurement.d_spread_px, 0.0)
        self.assertIsNone(measurement.bbox)
        self.assertGreater(measurement.area_px2, 0)

    def test_component_within_band_above_line_is_measured(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:48, 20:80] = 1  # Bottom row 47: 3 px above the line, inside the band.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            reflection_mode="none",
            surface_band_px=3,
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, 0.0), config)

        self.assertEqual(measurement.d_spread_px, 60.0)

    def test_fragmented_surface_intersection_measures_outer_span(self) -> None:
        # A rebounding ring can lift off the line mid-span; the diameter is
        # the outer width of the silhouette on the line, not the longest
        # unbroken contact segment.
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:40, 20:80] = 1  # Connected body above the line.
        mask[40:51, 20:35] = 1  # Left lobe reaching the line.
        mask[40:51, 65:80] = 1  # Right lobe reaching the line.
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, 0.0), config)

        self.assertEqual(measurement.d_spread_px, 60.0)


class SurfaceCalibrationTests(unittest.TestCase):
    def test_nonreflection_calibration_uses_droplet_lower_contour(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            background = np.full((100, 120), 4000, dtype=np.uint16)
            background[75:] = 2000
            calibration = background.copy()
            calibration[30:56, 35:85] = 1000
            # A larger change clipped by the image border must not be selected
            # instead of the complete impacted-droplet contour.
            calibration[85:, :100] = 0
            background_path = input_dir / "capture_000001.tif"
            calibration_path = input_dir / "capture_000002.tif"
            tifffile.imwrite(background_path, background)
            tifffile.imwrite(calibration_path, calibration)
            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                min_droplet_area_px2=10,
                morphology_radius_px=1,
                debug_dir=None,
            )

            line = estimate_surface_line_from_nonreflection_frame(
                [background_path, calibration_path],
                background.astype(np.float32),
                2,
                1000,
                config,
                make_structure(1),
            )

        self.assertAlmostEqual(line.center_y, 55.0, delta=1.0)
        self.assertAlmostEqual(line.angle_deg, 0.0, delta=0.2)

    def test_contact_line_uses_outward_tips_without_reflection_lobes(self) -> None:
        component = np.zeros((100, 180), dtype=bool)
        row_ys = np.arange(20, 81)
        left = 25 + np.abs(row_ys - 52)
        right = 155 - np.abs(row_ys - 49)
        for row_y, left_x, right_x in zip(row_ys, left, right):
            component[row_y, left_x : right_x + 1] = True

        line = estimate_contact_line(component)

        expected_slope = (49 - 52) / (155 - 25)
        self.assertAlmostEqual(line.slope, expected_slope, places=3)

    def test_contact_line_uses_notches_between_reflection_lobes(self) -> None:
        component = np.zeros((100, 180), dtype=bool)
        row_ys = np.arange(20, 81)
        left = np.interp(row_ys, [20, 35, 50, 65, 80], [55, 25, 40, 25, 55])
        right = np.interp(
            row_ys,
            [20, 37, 52, 67, 80],
            [115, 155, 140, 155, 115],
        )
        for row_y, left_x, right_x in zip(row_ys, left, right):
            component[row_y, int(round(left_x)) : int(round(right_x)) + 1] = True

        line = estimate_contact_line(component)

        expected_slope = (52 - 50) / (140 - 40)
        self.assertAlmostEqual(line.slope, expected_slope, places=3)

    def test_symmetric_component_can_be_selected_above_wrong_coarse_edge(self) -> None:
        mask = np.zeros((100, 120), dtype=np.uint8)
        for offset in range(-20, 21):
            half_width = 25 - abs(offset) // 2
            mask[40 + offset, 60 - half_width : 61 + half_width] = 1
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=100,
            morphology_radius_px=1,
            debug_dir=None,
        )

        strict_selection = select_calibration_component(mask, 70, config)
        symmetric_selection = select_symmetric_calibration_component(mask, 70, config)

        self.assertIsNone(strict_selection)
        self.assertIsNotNone(symmetric_selection)
        assert symmetric_selection is not None
        component, bbox = symmetric_selection
        surface_y_px, symmetry_error = estimate_vertical_symmetry_y(component, bbox)
        self.assertEqual(surface_y_px, 40)
        self.assertAlmostEqual(symmetry_error, 0.0)


if __name__ == "__main__":
    unittest.main()
