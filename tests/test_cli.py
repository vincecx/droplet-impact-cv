from __future__ import annotations

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from droplet_impact_cv.cli import build_parser, config_from_args, main, parse_args
from droplet_impact_cv.models import (
    FrameMeasurement,
    DEFAULT_MIN_STRONG_THRESHOLD,
    DEFAULT_PIXEL_SIZE_MM_PER_PX,
)


class CliConfigTests(unittest.TestCase):
    def test_cli_summary_reports_both_reference_methods(self) -> None:
        measurement = FrameMeasurement(
            frame_number=5, filename="frame000005.tif", time_ms=0.0,
            d_spread_px=0.0, d_spread_mm=0.0, droplet_area_px2=1200,
            surface_y_px=80, impact_frame=None, fps=8000.0,
            pixel_size_mm_per_px=0.01, calibration_frame=None,
            ref_frame=4, ref_d_eq_volume_mm=0.36,
            ref_d_eq_area_mm=0.39,
        )
        output = StringIO()
        with patch("droplet_impact_cv.cli.analyze_sequence", return_value=[measurement]), \
                patch("droplet_impact_cv.cli.write_csv"), redirect_stdout(output):
            main(["--include-pre-impact"])
        self.assertIn("Ref d_eq_volume: 0.360000 mm at frame 4", output.getvalue())
        self.assertIn("Ref d_eq_area: 0.390000 mm at frame 4", output.getvalue())

    def test_input_folder_config_overrides_code_defaults(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "# Experiment settings\n"
                "--fps 4000\n"
                "--pixel-size-mm-per-px 0.01682736321\n"
                "--calibration-frame 9105\n"
                "--reflection-mode none\n",
                encoding="utf-8",
            )

            config = config_from_args(parse_args([str(input_dir)]))

        self.assertEqual(config.fps, 4000.0)
        self.assertEqual(config.pixel_size_mm_per_px, 0.01682736321)
        self.assertEqual(config.calibration_frame, 9105)
        self.assertEqual(config.reflection_mode, "none")
        self.assertEqual(config.min_strong_threshold, DEFAULT_MIN_STRONG_THRESHOLD)

    def test_command_line_overrides_input_folder_config(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "--fps 4000\n"
                "--pixel-size-mm-per-px 0.01682736321\n"
                "--include-pre-impact\n",
                encoding="utf-8",
            )

            config = config_from_args(
                parse_args(
                    [
                        str(input_dir),
                        "--fps",
                        "2000",
                        "--no-include-pre-impact",
                    ]
                )
            )

        self.assertEqual(config.fps, 2000.0)
        self.assertEqual(config.pixel_size_mm_per_px, 0.01682736321)
        self.assertFalse(config.include_pre_impact)

    def test_missing_config_file_uses_code_defaults(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            config = config_from_args(parse_args([temporary_dir]))

        self.assertEqual(config.fps, 8000.0)
        self.assertEqual(config.pixel_size_mm_per_px, DEFAULT_PIXEL_SIZE_MM_PER_PX)
        self.assertFalse(config.include_pre_impact)

    def test_input_specific_default_output_paths(self) -> None:
        args = build_parser().parse_args(["sourcedata/example"])
        config = config_from_args(args)

        self.assertEqual(config.output_csv, Path("outputs/example/spreading_diameter.csv"))
        self.assertEqual(config.debug_dir, Path("outputs/example/debug_overlays"))

    def test_frame_range_is_optional_and_inclusive_limits_are_stored(self) -> None:
        parser = build_parser()
        default_config = config_from_args(parser.parse_args([]))
        bounded_config = config_from_args(
            parser.parse_args(["--start-frame", "12", "--end-frame", "62"])
        )

        self.assertIsNone(default_config.start_frame)
        self.assertIsNone(default_config.end_frame)
        self.assertEqual(bounded_config.start_frame, 12)
        self.assertEqual(bounded_config.end_frame, 62)

    def test_frame_range_limits_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--start-frame", "0"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--end-frame", "0"])

    def test_default_foreground_delta_rejects_weak_shadow(self) -> None:
        config = config_from_args(build_parser().parse_args([]))
        self.assertEqual(config.min_strong_threshold, DEFAULT_MIN_STRONG_THRESHOLD)
        self.assertEqual(config.min_strong_threshold, 1500.0)

    def test_surface_angle_is_automatic_unless_explicitly_overridden(self) -> None:
        parser = build_parser()
        automatic = config_from_args(parser.parse_args(["--calibration-frame", "189"]))
        overridden = config_from_args(
            parser.parse_args(
                ["--calibration-frame", "189", "--surface-angle-deg", "1.25"]
            )
        )

        self.assertIsNone(automatic.surface_angle_deg)
        self.assertEqual(overridden.surface_angle_deg, 1.25)

    def test_canonical_detection_parameter_names_map_to_analysis_config(self) -> None:
        config = config_from_args(
            build_parser().parse_args(
                [
                    "--surface-y-px",
                    "800",
                    "--strong-threshold",
                    "1800",
                    "--weak-threshold",
                    "450",
                    "--weak-threshold-ratio",
                    "0.3",
                    "--weak-reach-px",
                    "32",
                    "--min-strong-threshold",
                    "1200",
                    "--min-droplet-area-px2",
                    "300",
                    "--foreground-depth-px",
                    "200",
                    "--min-contact-px",
                    "25",
                    "--debug-interval",
                    "5",
                ]
            )
        )

        self.assertEqual(config.surface_y_px, 800)
        self.assertEqual(config.strong_threshold, 1800.0)
        self.assertEqual(config.weak_threshold, 450.0)
        self.assertEqual(config.weak_threshold_ratio, 0.3)
        self.assertEqual(config.weak_reach_px, 32)
        self.assertEqual(config.min_strong_threshold, 1200.0)
        self.assertEqual(config.min_droplet_area_px2, 300)
        self.assertEqual(config.foreground_depth_px, 200)
        self.assertEqual(config.min_contact_px, 25)
        self.assertEqual(config.debug_interval, 5)

    def test_canonical_unit_names_map_to_analysis_config(self) -> None:
        config = config_from_args(build_parser().parse_args([
            "--pixel-size-mm-per-px", "0.01", "--min-droplet-area-px2", "400",
            "--min-contact-px", "20", "--hydrophobic", "1",
        ]))
        self.assertEqual(config.pixel_size_mm_per_px, 0.01)
        self.assertEqual(config.min_droplet_area_px2, 400)
        self.assertEqual(config.min_contact_px, 20)
        self.assertTrue(config.hydrophobic)

    def test_old_config_options_are_rejected_even_with_current_cli_overrides(self) -> None:
        options = (
            ("--pixel-size-mm", "--pixel-size-mm-per-px"),
            ("--min-droplet-area-px", "--min-droplet-area-px2"),
            ("--min-contact-pixels", "--min-contact-px"),
            ("--HO", "--hydrophobic"),
        )
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            for old_option, current_option in options:
                (input_dir / "cv_config.txt").write_text(
                    f"{old_option} 1\n", encoding="utf-8"
                )
                for cli_options in ([], [current_option, "1"]):
                    with self.subTest(option=old_option, cli_options=cli_options):
                        errors = StringIO()
                        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                            parse_args([str(input_dir), *cli_options])
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn(f"unrecognized arguments: {old_option}", errors.getvalue())

    def test_debug_interval_defaults_to_one(self) -> None:
        config = config_from_args(build_parser().parse_args([]))

        self.assertEqual(config.debug_interval, 1)

    def test_hydrophobic_option_can_be_loaded_from_config_or_overridden_by_cli(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text("--hydrophobic 1\n", encoding="utf-8")

            configured = config_from_args(parse_args([str(input_dir)]))
            overridden = config_from_args(parse_args([str(input_dir), "--hydrophobic", "0"]))

        self.assertTrue(configured.hydrophobic)
        self.assertFalse(overridden.hydrophobic)

    def test_hydrophobic_option_must_be_zero_or_one(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--hydrophobic", "2"])

    def test_renamed_parameters_are_rejected(self) -> None:
        parser = build_parser()
        old_options = (
            "--output",
            "--surface-y",
            "--surface-frame",
            "--threshold",
            "--weak-threshold-fraction",
            "--weak-propagation-px",
            "--min-foreground-delta",
            "--min-area-px",
            "--measure-below-surface-px",
            "--touch-above-surface-px",
            "--touch-below-surface-px",
            "--min-touch-pixels",
            "--debug-every",
        )

        for option in old_options:
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([option, "1"])

    def test_deprecated_contact_window_parameters_are_rejected(self) -> None:
        parser = build_parser()
        for option in ("--contact-above-px", "--contact-below-px"):
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([option, "1"])

    def test_command_line_overrides_configured_reflection_mode(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "--reflection-mode mirror\n",
                encoding="utf-8",
            )

            config = config_from_args(
                parse_args([str(input_dir), "--reflection-mode", "none"])
            )

        self.assertEqual(config.reflection_mode, "none")


if __name__ == "__main__":
    unittest.main()
