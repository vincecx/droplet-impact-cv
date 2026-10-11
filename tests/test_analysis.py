from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import tifffile

from droplet_impact_cv.analysis import analyze_sequence
from droplet_impact_cv.imaging import build_background, component_measurement
from droplet_impact_cv.models import AnalysisConfig, SurfaceLine


class SequenceAnalysisTests(unittest.TestCase):
    def test_frame_range_limits_sequence_inclusively(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in range(1, 4):
                tifffile.imwrite(input_dir / f"frame{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=50,
                start_frame=2,
                end_frame=3,
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [2, 3])
        for row in measurements:
            self.assertIsNone(row.ref_frame)
            self.assertIsNone(row.ref_d_eq_volume_px)
            self.assertIsNone(row.ref_d_eq_area_px)
            self.assertIsNone(row.ref_d_eq_area_mm)

    def test_frame_numbers_come_from_six_digit_filename_suffixes(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in (5, 6, 8):
                tifffile.imwrite(input_dir / f"capture_{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=50,
                include_pre_impact=True,
                time_zero="first-frame",
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [5, 6, 8])
        self.assertEqual([row.time_ms for row in measurements], [0.0, 0.125, 0.375])

    def test_frame_range_uses_filename_frame_number(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in (5, 6, 7):
                tifffile.imwrite(input_dir / f"capture_{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=50,
                start_frame=6,
                end_frame=6,
                debug_dir=None,
            )
            measurements = analyze_sequence(config)

        self.assertEqual([row.frame_number for row in measurements], [6])

    def test_start_frame_must_not_exceed_end_frame(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            tifffile.imwrite(input_dir / "capture_000005.tif", image)
            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                start_frame=7,
                end_frame=6,
                debug_dir=None,
            )

            with self.assertRaisesRegex(
                ValueError, "start_frame must be less than or equal to end_frame"
            ):
                analyze_sequence(config)

    def test_start_frame_selects_background_frame(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in range(1, 4):
                tifffile.imwrite(input_dir / f"capture_{frame_number:06d}.tif", image)
            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=50,
                start_frame=2,
                debug_dir=None,
            )

            with patch(
                "droplet_impact_cv.analysis.build_background",
                wraps=build_background,
            ) as build_background_mock:
                analyze_sequence(config)

        background_files = build_background_mock.call_args.args[0]
        self.assertEqual(
            [path.name for path in background_files],
            ["capture_000002.tif", "capture_000003.tif"],
        )


class DebugOverlayLifecycleTests(unittest.TestCase):
    def test_stale_overlays_from_previous_runs_are_removed(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            debug_dir = input_dir / "debug_overlays"
            debug_dir.mkdir()
            stale_overlay = debug_dir / "000099.png"
            stale_overlay.write_bytes(b"stale")
            unrelated_file = debug_dir / "notes.txt"
            unrelated_file.write_text("keep me")
            image = np.full((100, 100), 4000, dtype=np.uint16)
            image[50:] = 2000
            for frame_number in range(1, 4):
                tifffile.imwrite(input_dir / f"frame{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=50,
                debug_dir=debug_dir,
            )
            analyze_sequence(config)

            self.assertFalse(stale_overlay.exists())
            self.assertTrue(unrelated_file.exists())

    def test_reference_diameter_is_frozen_when_clearance_reaches_half_diameter(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            debug_dir = input_dir / "debug_overlays"
            background = np.full((120, 140), 4000, dtype=np.uint16)
            background[80:] = 2000
            tifffile.imwrite(input_dir / "frame000001.tif", background)
            for frame_number, top in ((2, 20), (3, 30), (4, 34), (5, 40)):
                image = background.copy()
                right = 90 if frame_number == 5 else 70
                image[top : top + 30, 30:right] = 2000
                tifffile.imwrite(input_dir / f"frame{frame_number:06d}.tif", image)

            config = AnalysisConfig(
                input_dir=input_dir,
                output_csv=input_dir / "output.csv",
                surface_y_px=80,
                surface_angle_deg=0.0,
                morphology_radius_px=0,
                min_droplet_area_px2=1,
                min_contact_px=1,
                include_pre_impact=True,
                debug_dir=debug_dir,
                debug_interval=100,
            )
            with patch("droplet_impact_cv.analysis.write_debug_overlay") as overlay:
                measurements = analyze_sequence(config)

        self.assertEqual([row.ref_frame for row in measurements], [4, 4, 4, 4, 4])
        self.assertTrue(
            all(row.ref_d_eq_volume_px is not None for row in measurements)
        )
        for row in measurements:
            self.assertAlmostEqual(row.ref_d_eq_volume_px, (40**2 * 30)**(1/3))
            self.assertAlmostEqual(row.ref_d_eq_area_px, math.sqrt(4 * 1200 / math.pi))
            self.assertAlmostEqual(
                row.ref_d_eq_area_mm,
                row.ref_d_eq_area_px * config.pixel_size_mm_per_px,
            )
            self.assertAlmostEqual(
                row.ref_d_eq_volume_mm,
                row.ref_d_eq_volume_px * config.pixel_size_mm_per_px,
            )
        self.assertNotEqual(measurements[-1].droplet_area_px2, measurements[-2].droplet_area_px2)
        self.assertEqual(overlay.call_count, 2)
        self.assertIsNotNone(overlay.call_args_list[0].args[2])
        self.assertIsNotNone(overlay.call_args_list[1].args[2])


class ContourMeasurementTests(unittest.TestCase):
    def test_measurement_never_adds_pixels_outside_detected_contour(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[30:55, 20:80] = 1
        mask[45:55, 20:30] = 0
        config = AnalysisConfig(
            input_dir=Path("input"),
            output_csv=Path("output.csv"),
            min_droplet_area_px2=1,
            min_contact_px=1,
            debug_dir=None,
        )

        measurement = component_measurement(mask, SurfaceLine(50.0, angle_deg=0.0), config)

        self.assertFalse(np.any(measurement.mask & ~mask.astype(bool)))
        self.assertEqual(measurement.d_spread_px, 50.0)


if __name__ == "__main__":
    unittest.main()
