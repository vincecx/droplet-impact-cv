from __future__ import annotations

import csv
from dataclasses import replace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from droplet_impact_cv.models import FrameMeasurement, SurfaceLine
from droplet_impact_cv.output import write_csv
from droplet_impact_cv.visualization import write_debug_overlay


class OutputTests(unittest.TestCase):
    def test_debug_overlay_uses_antialiased_hershey_duplex(self) -> None:
        image = np.zeros((80, 100), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        with TemporaryDirectory() as temporary_dir, patch(
            "droplet_impact_cv.visualization.cv2.putText", wraps=cv2.putText
        ) as put_text:
            output_path = Path(temporary_dir) / "overlay.png"
            write_debug_overlay(
                output_path,
                image,
                mask,
                SurfaceLine(50.0, angle_deg=0.0),
                None,
                1,
                1.0,
                d_contact_mm=0.5,
            )

        self.assertEqual(put_text.call_args.args[3], cv2.FONT_HERSHEY_DUPLEX)
        self.assertEqual(put_text.call_args.args[7], cv2.LINE_AA)
        self.assertEqual(put_text.call_args.args[4], 0.7)
        self.assertNotIn("D0", put_text.call_args.args[1])

    def test_rim_to_rim_overlay_draws_contact_and_rim_measurements(self) -> None:
        image = np.zeros((80, 100), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "overlay.png"
            write_debug_overlay(
                output_path,
                image,
                mask,
                SurfaceLine(50.0, angle_deg=0.0),
                (20, 50, 80, 50),
                1,
                1.0,
                rim_bbox=(10, 30, 90, 30),
                d_contact_mm=0.5,
                d_rim_mm=1.0,
            )
            overlay = cv2.imread(str(output_path))

        self.assertTrue(np.array_equal(overlay[50, 20], np.array([0, 255, 0])))
        self.assertTrue(np.array_equal(overlay[30, 40], np.array([255, 0, 255])))

    def test_csv_columns_and_analysis_parameters(self) -> None:
        measurement = FrameMeasurement(
            frame_number=61,
            filename="frame0061.tif",
            time_ms=0.0,
            d_spread_px=156.004,
            d_spread_mm=1.110565076,
            droplet_area_px2=72128,
            surface_y_px=798,
            impact_frame=61,
            fps=8000.0,
            pixel_size_mm_per_px=0.00711883341,
            surface_calibration_frame=61,
            d_contact_px=140.0,
            d_contact_mm=0.9966366774,
            d_rim_px=156.004,
            d_rim_mm=1.110565076,
            droplet_size_ref_frame=42,
            ref_d_eq_volume_px=150.0,
            ref_d_eq_volume_mm=1.0678250115,
            ref_d_eq_area_px=148.0,
            ref_d_eq_area_mm=1.05358734468,
            ref_d_parallel_px=156.004,
            ref_d_parallel_mm=1.110565076,
            ref_d_normal_px=139.0,
            ref_d_normal_mm=0.989517864,
            ref_clearance_px=74.0,
            ref_clearance_mm=0.526793672,
            shadow_removed_px2=123,
            segmentation_warning="Check contact boundary.",
        )
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "result.csv"
            write_csv(output_path, [measurement])
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows[0],
            [
                "filename",
                "frame_number",
                "time_ms",
                "d_spread_px",
                "d_spread_mm",
                "d_contact_px",
                "d_contact_mm",
                "d_rim_px",
                "d_rim_mm",
                "droplet_size_ref_frame",
                "ref_d_eq_volume_px",
                "ref_d_eq_volume_mm",
                "ref_d_eq_area_px",
                "ref_d_eq_area_mm",
                "ref_d_parallel_px",
                "ref_d_parallel_mm",
                "ref_d_normal_px",
                "ref_d_normal_mm",
                "ref_clearance_px",
                "ref_clearance_mm",
                "impact_frame",
                "surface_y_px",
                "droplet_area_px2",
                "fps",
                "pixel_size_mm_per_px",
                "surface_calibration_frame",
                "shadow_removed_px2",
                "segmentation_warning",
            ],
        )
        self.assertEqual(
            rows[1],
            [
                "frame0061.tif",
                "61",
                "0.000000",
                "156.004",
                "1.110565076",
                "140.000",
                "0.996636677",
                "156.004",
                "1.110565076",
                "42",
                "150.000",
                "1.067825012",
                "148.000",
                "1.053587345",
                "156.004",
                "1.110565076",
                "139.000",
                "0.989517864",
                "74.000",
                "0.526793672",
                "61",
                "798",
                "72128",
                "8000.0",
                "0.00711883341",
                "61",
                "123",
                "Check contact boundary.",
            ],
        )

    def test_csv_leaves_missing_reference_diameters_empty(self) -> None:
        measurement = FrameMeasurement(
            frame_number=1, filename="frame000001.tif", time_ms=0.0,
            d_spread_px=0.0, d_spread_mm=0.0, droplet_area_px2=0,
            surface_y_px=80, impact_frame=None, fps=8000.0,
            pixel_size_mm_per_px=0.01, surface_calibration_frame=None,
        )
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "result.csv"
            write_csv(output_path, [measurement, replace(measurement, ref_d_eq_area_px=float("nan"))])
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        for row in rows:
            for column in (
                "droplet_size_ref_frame", "ref_d_eq_volume_px", "ref_d_eq_volume_mm",
                "ref_d_eq_area_px", "ref_d_eq_area_mm",
            ):
                self.assertEqual(row[column], "")


if __name__ == "__main__":
    unittest.main()
