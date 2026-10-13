from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from droplet_impact_cv.models import AnalysisConfig, FrameMeasurement
from droplet_impact_cv.output import write_csv, write_run_params
from scripts.plot_figures import load_run, main


class PlotFiguresTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.csv = self.folder / "cv_output.csv"
        self.config = AnalysisConfig(input_dir=self.folder, output_dir=self.folder,
                                     output_csv=self.csv, surface_angle_deg=12,
                                     debug_dir=self.folder / "debug_overlays")
        rows = [FrameMeasurement(
            frame_number=index + 1, filename=f"frame{index + 1:06d}.tif",
            time_ms=index * 0.5, d_spread_px=diameter * 100, d_spread_mm=diameter,
            droplet_area_px2=3000, surface_y_px=80, impact_frame=1, fps=2000,
            pixel_size_mm_per_px=0.01, surface_calibration_frame=3,
            d_contact_mm=diameter, d_rim_mm=diameter,
            droplet_size_ref_frame=1, ref_d_eq_volume_mm=1, ref_d_eq_area_mm=1,
            ref_d_parallel_mm=1, ref_d_normal_mm=1, ref_clearance_mm=0.1,
        ) for index, diameter in enumerate((1, 2, 1.5))]
        write_csv(self.csv, rows)
        write_run_params(self.folder / "run_params.json", self.config)

    def test_load_current_output_folder_and_export_figures(self):
        run = load_run(self.folder)
        self.assertEqual(run.csv_path, self.csv)
        self.assertEqual(run.surface_angle_deg, 12)
        self.assertEqual(run.n_rows, 3)
        with redirect_stdout(StringIO()):
            result = main([str(self.folder), "--formats", "png", "--dpi", "40"])
        self.assertEqual(result, 0)
        figures = self.folder / "figures"
        self.assertTrue((figures / "spreading_metrics_summary.csv").is_file())
        pngs = list(figures.glob("*.png"))
        self.assertTrue(pngs)
        self.assertTrue(all(path.stat().st_size > 1000 for path in pngs))

    def test_reject_former_filename_and_partial_csv(self):
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            load_run(self.folder / "spreading_diameter.csv")
        self.csv.write_text("time_ms,d_spread_mm\n0,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing column"):
            load_run(self.csv)

    def test_reject_old_run_params(self):
        path = self.folder / "run_params.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["parameters"]["output_dir"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a current-format run parameters file"):
            load_run(self.csv)


if __name__ == "__main__":
    unittest.main()
