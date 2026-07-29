from __future__ import annotations

import csv
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
                contact_line_diameter_mm=0.5,
            )

        self.assertEqual(put_text.call_args.args[3], cv2.FONT_HERSHEY_DUPLEX)
        self.assertEqual(put_text.call_args.args[7], cv2.LINE_AA)
        self.assertEqual(put_text.call_args.args[4], 0.7)
        self.assertNotIn("D0", put_text.call_args.args[1])

    def test_hydrophobic_overlay_draws_contact_and_rim_measurements(self) -> None:
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
                contact_line_diameter_mm=0.5,
                rim_to_rim_diameter_mm=1.0,
            )
            overlay = cv2.imread(str(output_path))

        self.assertTrue(np.array_equal(overlay[50, 20], np.array([0, 255, 0])))
        self.assertTrue(np.array_equal(overlay[30, 40], np.array([255, 0, 255])))

    def test_csv_columns_and_analysis_parameters(self) -> None:
        measurement = FrameMeasurement(
            frame_number=61,
            filename="frame0061.tif",
            time_ms=0.0,
            spreading_diameter_px=156.004,
            spreading_diameter_mm=1.110565076,
            component_area_px=72128,
            surface_y=798,
            impact_frame=61,
            fps=8000.0,
            pixel_size_mm=0.00711883341,
            surface_frame=61,
            contact_line_diameter_px=140.0,
            contact_line_diameter_mm=0.9966366774,
            rim_to_rim_diameter_px=156.004,
            rim_to_rim_diameter_mm=1.110565076,
            reference_frame=42,
            initial_equivalent_diameter_px=150.0,
            initial_equivalent_diameter_mm=1.0678250115,
            initial_surface_parallel_diameter_px=156.004,
            initial_surface_parallel_diameter_mm=1.110565076,
            initial_surface_normal_diameter_px=139.0,
            initial_surface_normal_diameter_mm=0.989517864,
            surface_clearance_px=74.0,
            surface_clearance_mm=0.526793672,
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
                "frame",
                "time_ms",
                "spreading_diameter_px",
                "spreading_diameter_mm",
                "contact_line_diameter_px",
                "contact_line_diameter_mm",
                "rim_to_rim_diameter_px",
                "rim_to_rim_diameter_mm",
                "reference_frame",
                "initial_equivalent_diameter_px",
                "initial_equivalent_diameter_mm",
                "initial_surface_parallel_diameter_px",
                "initial_surface_parallel_diameter_mm",
                "initial_surface_normal_diameter_px",
                "initial_surface_normal_diameter_mm",
                "surface_clearance_px",
                "surface_clearance_mm",
                "impact_frame",
                "surface_y_px",
                "component_area_px",
                "fps",
                "pixel-size-mm",
                "surface-frame",
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
            ],
        )


if __name__ == "__main__":
    unittest.main()
