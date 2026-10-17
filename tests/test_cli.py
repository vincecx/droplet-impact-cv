from __future__ import annotations

import unittest
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from droplet_impact_cv.cli import PROJECT_DIR, build_parser, config_from_args, default_output_dir, main, parse_args
from droplet_impact_cv.models import (
    FrameMeasurement,
    DEFAULT_MIN_STRONG_CHANGE_THRESHOLD,
    DEFAULT_PIXEL_SIZE_MM_PER_PX,
)


class CliConfigTests(unittest.TestCase):
    def test_gas_area_config_and_cli_precedence(self):
        parser = build_parser()
        self.assertEqual(config_from_args(parser.parse_args(["--min-gas-area-px2", "42"])).min_gas_area_px2, 42)
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text('--min-gas-area-px2 30\n', encoding="utf-8")
            config = config_from_args(parse_args([str(folder), "--min-gas-area-px2", "42"]))
            self.assertEqual(config.min_gas_area_px2, 42)
            (folder / "cv_config.txt").write_text('--min-gas-area-px2 42\n', encoding="utf-8")
            config = config_from_args(parse_args([str(folder), "--min-gas-area-px2", "30"]))
            self.assertEqual(config.min_gas_area_px2, 30)

    def test_superseded_parameter_names_are_rejected(self):
        options = ["--min-contact-px", "--weak-reach-px", "--surface-band-px",
                   "--reflection-mode", "--shadow-suppression", "--no-shadow-suppression",
                   "--min-droplet-area-px2", "--min-strong-threshold",
                   "--strong-threshold", "--weak-threshold", "--weak-threshold-ratio"]
        for option in options:
            arguments = [option] if "shadow-suppression" in option else [option, "1"]
            with self.subTest(option=option), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(arguments)

    def test_approach_shadow_guard_config_and_cli_override(self):
        self.assertTrue(config_from_args(build_parser().parse_args([])).approach_shadow_guard)
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text('--no-approach-shadow-guard\n', encoding="utf-8")
            self.assertFalse(config_from_args(parse_args([str(folder)])).approach_shadow_guard)
            self.assertTrue(config_from_args(parse_args([str(folder), "--approach-shadow-guard"])).approach_shadow_guard)

    def test_short_output_option_selects_directory_in_cli_and_folder_config(self):
        # A CSV-looking argument is still a directory; -o has no CSV alias.
        config = config_from_args(build_parser().parse_args(["-o", "results.csv"]))
        self.assertEqual(config.output_dir, Path("results.csv"))
        self.assertEqual(config.output_csv, Path("results.csv/cv_output.csv"))
        self.assertEqual(config.debug_dir, Path("results.csv/debug_overlays"))
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text('-o "saved folder"\n', encoding="utf-8")
            saved = config_from_args(parse_args([str(folder)]))
            self.assertEqual(saved.output_dir, Path("saved folder"))
            overridden = config_from_args(parse_args([
                str(folder), "-o", "cli folder", "--output-csv", "custom.csv",
            ]))
            self.assertEqual(overridden.output_dir, Path("cli folder"))
            self.assertEqual(overridden.output_csv, Path("custom.csv"))

    def test_output_dir_layout_and_legacy_override_precedence(self):
        parser = build_parser()
        folder = Path("new output")
        config = config_from_args(parser.parse_args(["--output-dir", str(folder)]))
        self.assertEqual(config.output_dir, folder)
        self.assertEqual(config.output_csv, folder / "cv_output.csv")
        self.assertEqual(config.debug_dir, folder / "debug_overlays")
        for tokens in (["--output-dir", str(folder), "--output-csv", "other/custom.csv", "--debug-dir", "other/png"],
                       ["--debug-dir", "other/png", "--output-csv", "other/custom.csv", "--output-dir", str(folder)]):
            config = config_from_args(parser.parse_args(tokens))
            self.assertEqual(config.output_dir, folder)
            self.assertEqual(config.output_csv, Path("other/custom.csv"))
            self.assertEqual(config.debug_dir, Path("other/png"))

    def test_folder_output_options_and_cli_override_precedence(self):
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text(
                '--output-dir "saved folder"\n--output-csv "legacy/custom.csv"\n'
                '--debug-dir "legacy/png"\n', encoding="utf-8",
            )
            config = config_from_args(parse_args([str(folder), "--output-dir", "cli folder"]))
            self.assertEqual(config.output_dir, Path("cli folder"))
            self.assertEqual(config.output_csv, Path("legacy/custom.csv"))
            self.assertEqual(config.debug_dir, Path("legacy/png"))
            config = config_from_args(parse_args([str(folder), "--output-csv", "cli/custom.csv"]))
            self.assertEqual(config.output_dir, Path("saved folder"))
            self.assertEqual(config.output_csv, Path("cli/custom.csv"))

    def test_output_dir_writes_fixed_csv_and_run_params_even_with_legacy_override(self):
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            output = folder / "results"
            csv = folder / "legacy" / "custom.csv"
            with patch("droplet_impact_cv.cli.analyze_sequence", return_value=[]), redirect_stdout(StringIO()):
                main([str(folder), "--output-dir", str(output)])
            self.assertTrue((output / "cv_output.csv").is_file())
            self.assertFalse((output / "spreading_diameter.csv").exists())
            with patch("droplet_impact_cv.cli.analyze_sequence", return_value=[]), redirect_stdout(StringIO()):
                main([str(folder), "--output-dir", str(output), "--output-csv", str(csv)])
            self.assertTrue(csv.is_file())
            self.assertFalse((csv.parent / "run_params.json").exists())
            params = json.loads((output / "run_params.json").read_text(encoding="utf-8"))["parameters"]
            self.assertEqual(params["output_dir"], str(output))
            self.assertEqual(params["output_csv"], str(csv))
            self.assertEqual(params["debug_dir"], str(output / "debug_overlays"))

    def test_spaced_unicode_folder_metadata_cli_override_and_json_output(self):
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir) / "2026.9.14 Al-60-5-SHO 最低高度 常温3"
            folder.mkdir()
            (folder / "cv_config.txt").write_text(
                '--substrate "Al surface 名称"\n--hydrophilicity SHO\n'
                '--substrate_temp_deg_c 24\n--rim-to-rim 1\n', encoding="utf-8",
            )
            config = config_from_args(parse_args([str(folder)]))
            self.assertEqual(config.input_dir, folder)
            self.assertEqual(config.output_csv.parent.name, folder.name)
            self.assertEqual(config.substrate, "Al surface 名称")
            self.assertEqual(config.hydrophilicity, "SHO")
            self.assertEqual(config.substrate_temp_deg_c, 24.0)
            self.assertTrue(config.rim_to_rim)
            output_csv = folder / "output folder" / "result.csv"
            with patch("droplet_impact_cv.cli.analyze_sequence", return_value=[]) as analyze, \
                    redirect_stdout(StringIO()):
                main([str(folder), "--output-csv", str(output_csv),
                      "--substrate", "Other surface", "--hydrophilicity", "HI",
                      "--substrate_temp_deg_c", "-5.5", "--rim-to-rim", "0"])
            payload = json.loads((output_csv.parent / "run_params.json").read_text(encoding="utf-8"))
            params = payload["parameters"]
            self.assertEqual(payload["package_version"], "0.5")
            self.assertEqual(params["input_dir"], str(folder))
            self.assertEqual(params["substrate"], "Other surface")
            self.assertEqual(params["hydrophilicity"], "HI")
            self.assertEqual(params["substrate_temp_deg_c"], -5.5)
            self.assertFalse(params["rim_to_rim"])
            self.assertNotIn("hydrophobic", params)
            self.assertEqual(analyze.call_args.args[0].substrate, params["substrate"])

    def test_metadata_defaults_and_wettability_choices(self):
        parser = build_parser()
        defaults = config_from_args(parser.parse_args([]))
        self.assertIsNone(defaults.substrate)
        self.assertIsNone(defaults.hydrophilicity)
        self.assertIsNone(defaults.substrate_temp_deg_c)
        for value in ("SHI", "HI", "HO", "SHO"):
            with self.subTest(value=value):
                self.assertEqual(parser.parse_args(["--hydrophilicity", value]).hydrophilicity, value)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--hydrophilicity", "invalid"])

    def test_hydrophobic_cli_option_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(["--hydrophobic", "1", "--rim-to-rim", "0"])
        self.assertEqual(raised.exception.code, 2)

    def test_top_exclusion_defaults_config_override_and_bounds(self):
        parser = build_parser()
        self.assertEqual(config_from_args(parser.parse_args([])).top_exclusion_ratio, 0.25)
        for value in ("0", "0.6", "1"):
            self.assertEqual(
                config_from_args(parser.parse_args(["--top-exclusion-ratio", value])).top_exclusion_ratio,
                float(value),
            )
        for value in ("-0.1", "1.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--top-exclusion-ratio", value])
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text("--top-exclusion-ratio 0.4\n", encoding="utf-8")
            self.assertEqual(config_from_args(parse_args([str(folder)])).top_exclusion_ratio, 0.4)
            self.assertEqual(config_from_args(parse_args([str(folder), "--top-exclusion-ratio", "0"])).top_exclusion_ratio, 0)

    def test_reference_and_surface_frames_are_independent(self):
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text(
                "--droplet-size-ref-frame 4\n--surface-calibration-frame 61\n", encoding="utf-8",
            )
            config = config_from_args(parse_args([str(folder)]))
            self.assertEqual(config.droplet_size_ref_frame, 4)
            self.assertEqual(config.surface_calibration_frame, 61)
            config = config_from_args(parse_args([
                str(folder), "--droplet-size-ref-frame", "5",
                "--surface-calibration-frame", "62",
            ]))
            self.assertEqual(config.droplet_size_ref_frame, 5)
            self.assertEqual(config.surface_calibration_frame, 62)
            automatic = config_from_args(build_parser().parse_args([]))
            self.assertIsNone(automatic.droplet_size_ref_frame)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["--droplet-size-ref-frame", "-1"])

    def test_legacy_calibration_frame_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(["--calibration-frame", "61"])
        self.assertEqual(error.exception.code, 2)
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            (folder / "cv_config.txt").write_text("--calibration-frame 61\n", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                parse_args([str(folder)])
            self.assertEqual(error.exception.code, 2)

    def test_mirror_depth_folder_config_cli_override_and_defaults(self):
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir)
            defaults = config_from_args(parse_args([str(folder)]))
            self.assertEqual(defaults.mirror_min_depth_px, 50)
            self.assertEqual(defaults.weak_change_threshold_ratio, 0.3)
            (folder / "cv_config.txt").write_text("--mirror-min-depth-px 25\n", encoding="utf-8")
            self.assertEqual(config_from_args(parse_args([str(folder)])).mirror_min_depth_px, 25)
            self.assertEqual(config_from_args(parse_args([str(folder), "--mirror-min-depth-px", "0"])).mirror_min_depth_px, 0)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args([str(folder), "--mirror-min-depth-px", "-1"])

    def test_cli_summary_reports_both_reference_methods(self) -> None:
        measurement = FrameMeasurement(
            frame_number=5, filename="frame000005.tif", time_ms=0.0,
            d_spread_px=0.0, d_spread_mm=0.0, droplet_area_px2=1200,
            surface_y_px=80, impact_frame=None, fps=8000.0,
            pixel_size_mm_per_px=0.01, surface_calibration_frame=None,
            droplet_size_ref_frame=4, ref_d_eq_volume_mm=0.36,
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
                "--surface-calibration-frame 9105\n"
                "--surface-calibration-method none\n",
                encoding="utf-8",
            )

            config = config_from_args(parse_args([str(input_dir)]))

        self.assertEqual(config.fps, 4000.0)
        self.assertEqual(config.pixel_size_mm_per_px, 0.01682736321)
        self.assertEqual(config.surface_calibration_frame, 9105)
        self.assertEqual(config.surface_calibration_method, "none")
        self.assertEqual(config.min_strong_change_threshold, DEFAULT_MIN_STRONG_CHANGE_THRESHOLD)

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

        self.assertEqual(config.fps, 1000.0)
        self.assertEqual(config.pixel_size_mm_per_px, DEFAULT_PIXEL_SIZE_MM_PER_PX)
        self.assertEqual(config.pixel_size_mm_per_px, 0.01)
        self.assertEqual(config.contact_band_half_height_px, 1)
        self.assertFalse(config.include_pre_impact)

    def test_input_specific_default_output_paths(self) -> None:
        args = build_parser().parse_args(["sourcedata/example"])
        config = config_from_args(args)

        self.assertEqual(config.output_csv, PROJECT_DIR / "outputs/example/cv_output.csv")
        self.assertEqual(config.debug_dir, PROJECT_DIR / "outputs/example/debug_overlays")

    def test_default_output_is_fixed_to_project_for_external_input_and_other_cwd(self) -> None:
        import os
        previous = Path.cwd()
        with TemporaryDirectory() as temporary_dir:
            folder = Path(temporary_dir) / "external-input"
            folder.mkdir()
            try:
                os.chdir(temporary_dir)
                config = config_from_args(build_parser().parse_args([str(folder)]))
                self.assertEqual(default_output_dir(folder), PROJECT_DIR / "outputs/external-input")
                self.assertEqual(config.output_csv, PROJECT_DIR / "outputs/external-input/cv_output.csv")
                self.assertEqual(config.debug_dir, PROJECT_DIR / "outputs/external-input/debug_overlays")
                self.assertTrue(config.output_csv.is_absolute())
            finally:
                os.chdir(previous)

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

    def test_default_auto_threshold_floor_allows_faint_liquid(self) -> None:
        config = config_from_args(build_parser().parse_args([]))
        self.assertEqual(config.min_strong_change_threshold, DEFAULT_MIN_STRONG_CHANGE_THRESHOLD)
        self.assertEqual(config.min_strong_change_threshold, 200.0)

    def test_surface_angle_is_automatic_unless_explicitly_overridden(self) -> None:
        parser = build_parser()
        automatic = config_from_args(parser.parse_args(["--surface-calibration-frame", "189"]))
        overridden = config_from_args(
            parser.parse_args(
                ["--surface-calibration-frame", "189", "--surface-angle-deg", "1.25"]
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
                    "--strong-change-threshold",
                    "1800",
                    "--weak-change-threshold",
                    "450",
                    "--weak-change-threshold-ratio",
                    "0.3",
                    "--weak-growth-steps",
                    "32",
                    "--min-strong-change-threshold",
                    "1200",
                    "--min-component-area-px2",
                    "300",
                    "--foreground-depth-px",
                    "200",
                    "--min-gas-area-px2",
                    "25",
                    "--debug-interval",
                    "5",
                ]
            )
        )

        self.assertEqual(config.surface_y_px, 800)
        self.assertEqual(config.strong_change_threshold, 1800.0)
        self.assertEqual(config.weak_change_threshold, 450.0)
        self.assertEqual(config.weak_change_threshold_ratio, 0.3)
        self.assertEqual(config.weak_growth_steps, 32)
        self.assertEqual(config.min_strong_change_threshold, 1200.0)
        self.assertEqual(config.min_component_area_px2, 300)
        self.assertEqual(config.foreground_depth_px, 200)
        self.assertEqual(config.min_gas_area_px2, 25)
        self.assertEqual(config.debug_interval, 5)

    def test_canonical_unit_names_map_to_analysis_config(self) -> None:
        config = config_from_args(build_parser().parse_args([
            "--pixel-size-mm-per-px", "0.01", "--min-component-area-px2", "400",
            "--min-gas-area-px2", "20", "--rim-to-rim", "1",
        ]))
        self.assertEqual(config.pixel_size_mm_per_px, 0.01)
        self.assertEqual(config.min_component_area_px2, 400)
        self.assertEqual(config.min_gas_area_px2, 20)
        self.assertTrue(config.rim_to_rim)

    def test_old_config_options_are_rejected_even_with_current_cli_overrides(self) -> None:
        options = (
            ("--pixel-size-mm", "--pixel-size-mm-per-px"),
            ("--min-droplet-area-px", "--min-component-area-px2"),
            ("--min-contact-pixels", "--min-gas-area-px2"),
            ("--HO", "--rim-to-rim"),
            ("--hydrophobic", "--rim-to-rim"),
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

    def test_rim_to_rim_option_can_be_loaded_from_config_or_overridden_by_cli(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text("--rim-to-rim 1\n", encoding="utf-8")

            configured = config_from_args(parse_args([str(input_dir)]))
            overridden = config_from_args(parse_args([str(input_dir), "--rim-to-rim", "0"]))

        self.assertTrue(configured.rim_to_rim)
        self.assertFalse(overridden.rim_to_rim)

    def test_rim_to_rim_option_must_be_zero_or_one(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--rim-to-rim", "2"])

    def test_renamed_parameters_are_rejected(self) -> None:
        parser = build_parser()
        old_options = (
            "--output",
            "--surface-y",
            "--surface-frame",
            "--threshold",
            "--weak-change-threshold-fraction",
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

    def test_command_line_overrides_configured_surface_calibration_method(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            (input_dir / "cv_config.txt").write_text(
                "--surface-calibration-method mirror\n",
                encoding="utf-8",
            )

            config = config_from_args(
                parse_args([str(input_dir), "--surface-calibration-method", "none"])
            )

        self.assertEqual(config.surface_calibration_method, "none")


if __name__ == "__main__":
    unittest.main()
