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
            )

        self.assertEqual(put_text.call_args.args[3], cv2.FONT_HERSHEY_DUPLEX)
        self.assertEqual(put_text.call_args.args[7], cv2.LINE_AA)

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
            )
            overlay = cv2.imread(str(output_path))

        self.assertTrue(np.array_equal(overlay[50, 20], np.array([0, 255, 0])))
        self.assertTrue(np.array_equal(overlay[30, 40], np.array([255, 0, 255])))

    def test_csv_columns_and_analysis_parameters(self) -> None:
        measurement = FrameMeasurement(
            frame_number=61,
            filename="frame0061.tif",
            time_ms=0.0,
            diameter_px=156.004,
            diameter_mm=1.110565076,
            component_area_px=72128,
            surface_y=798,
            impact_frame=61,
            fps=8000.0,
            pixel_size_mm=0.00711883341,
            surface_frame=61,
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
                "diameter_px",
                "diameter_mm",
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
