from __future__ import annotations

import gc
import sys
import time
import tkinter as tk
import unittest
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from droplet_impact_cv.analysis import analyze_sequence, prepare_sequence
from droplet_impact_cv.gui import (
    PARAMETERS, STATUS_COLORS, DropletApp, ParameterControl, PreviewSession, PreviewViewport,
    load_folder_config, resolve_output_config, save_folder_config, surface_values_from_points,
)
from droplet_impact_cv.cli import build_parser, default_output_dir
from droplet_impact_cv.models import CONFIG_PARAMETER_ORDER, AnalysisConfig
from droplet_impact_cv.output import run_params_payload
from droplet_impact_cv.imaging import read_image


class PreviewGeometryTests(unittest.TestCase):
    def test_zoom_keeps_cursor_image_position_and_ignores_letterbox(self):
        viewport = PreviewViewport()
        shape = (120, 140)
        point = viewport.image_point(300, 180, shape, 560, 400)
        viewport.zoom_at(300, 180, 2, shape, 560, 400)
        np.testing.assert_allclose(viewport.image_point(300, 180, shape, 560, 400), point)
        viewport.zoom_at(300, 180, 0.5, shape, 560, 400)
        self.assertAlmostEqual(viewport.center_x, 0.5)
        self.assertAlmostEqual(viewport.center_y, 0.5)
        viewport.zoom_at(0, 0, 2, shape, 560, 400)
        self.assertEqual(viewport.zoom, 1)

    def test_surface_center_and_clockwise_tilt_are_independent_of_point_order(self):
        points = [(20, 40), (80, 46)]
        values = surface_values_from_points(points, 101)
        self.assertEqual(values[0], 43)
        self.assertAlmostEqual(values[1], np.degrees(np.arctan(0.1)))
        self.assertEqual(surface_values_from_points(points[::-1], 101), values)
        with self.assertRaises(ValueError):
            surface_values_from_points([(20, 40), (20, 60)], 101)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "input 数据 with spaces"
        self.folder.mkdir()
        self.config_file = self.folder / "cv_config.txt"
        self.config_file.write_text(
            "--fps 4000\n--strong-threshold 1000\n--weak-threshold 500\n"
            "--surface-y-px 80\n--surface-angle-deg 0\n"
            "--min-droplet-area-px2 1\n--min-contact-px 1\n"
            "--morphology-radius-px 0\n--include-pre-impact\n",
            encoding="utf-8",
        )
        background = np.full((120, 140), 4000, dtype=np.uint16)
        background[80:] = 2000
        tifffile.imwrite(self.folder / "frame000001.tif", background)
        for number, top in ((2, 20), (3, 30), (4, 34), (5, 40)):
            image = background.copy()
            image[top:top + 30, 30:70] = 2000
            tifffile.imwrite(self.folder / f"frame{number:06d}.tif", image)

    def test_config_uses_folder_then_defaults_and_ignores_process_argv(self):
        with patch.object(sys, "argv", ["gui", "--fps", "999"]):
            config = load_folder_config(self.folder)
        self.assertEqual(config.fps, 4000)
        self.assertEqual(config.weak_reach_px, 40)

    def test_top_exclusion_preserves_background_calibration_and_matches_batch(self):
        config = replace(load_folder_config(self.folder), surface_calibration_frame=5, reflection_mode="none")
        full = prepare_sequence(replace(config, top_exclusion_ratio=0))
        excluded = prepare_sequence(replace(config, top_exclusion_ratio=1))
        np.testing.assert_array_equal(full.background, excluded.background)
        self.assertEqual(full.surface_line, excluded.surface_line)
        self.assertEqual(full.target_center_x, excluded.target_center_x)
        self.assertEqual(full.threshold, excluded.threshold)
        self.assertTrue(full.measure(full.files[-1])[1].mask.any())
        self.assertFalse(excluded.measure(excluded.files[-1])[1].mask.any())
        session = PreviewSession.open(self.folder)
        _, _, details = session.preview(1, {"top_exclusion_ratio": 1})
        self.assertIn("Area: 0 px2", details)
        self.assertFalse(session.prepared.measure(session.prepared.files[1])[1].mask.any())
        rows = analyze_sequence(replace(config, top_exclusion_ratio=1, debug_dir=None))
        self.assertTrue(all(row.droplet_area_px2 == 0 for row in rows))

    def test_output_choices_keep_debug_overlays_inside_selected_folder(self):
        config = replace(load_folder_config(self.folder),
                         output_csv=self.folder / "configured.csv", debug_dir=self.folder / "old-debug")
        choices = (
            ("automatic", "", default_output_dir(self.folder)),
            ("input", "", self.folder / "CV_postprocessing"),
            ("custom", str(self.folder / "manual-output"), self.folder / "manual-output"),
        )
        for mode, custom, folder in choices:
            with self.subTest(mode=mode):
                result = resolve_output_config(config, mode, custom)
                self.assertEqual(result.output_csv, folder / "cv_output.csv")
                self.assertEqual(result.output_dir, folder)
                self.assertEqual(result.debug_dir, folder / "debug_overlays")
                self.assertFalse(folder.exists())
        with self.assertRaises(ValueError):
            resolve_output_config(config, "custom", " ")
        with self.assertRaises(ValueError):
            resolve_output_config(config, "custom", str(self.config_file))

    def test_invalid_config_is_a_gui_error_instead_of_system_exit(self):
        self.config_file.write_text("--HO 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrecognized arguments"):
            load_folder_config(self.folder)

    def test_save_writes_all_output_options_in_shared_parameter_order(self):
        config = load_folder_config(self.folder)
        save_folder_config(config)
        lines = [line for line in self.config_file.read_text(encoding="utf-8").splitlines()
                 if line.startswith("--")]
        self.assertEqual([line.split()[0] for line in lines[:3]],
                         ["--output-dir", "--output-csv", "--debug-dir"])
        self.assertEqual(list(run_params_payload(config)["parameters"]), list(CONFIG_PARAMETER_ORDER))
        self.assertEqual(set(CONFIG_PARAMETER_ORDER), {field.name for field in fields(AnalysisConfig)})
        option_names = {action.dest: next(option for option in action.option_strings if option.startswith("--"))
                        for action in build_parser()._actions
                        if action.option_strings and action.dest != "help"}
        self.assertEqual([line.split()[0] for line in lines],
                         [option_names[name] for name in CONFIG_PARAMETER_ORDER
                          if name in option_names and getattr(config, name) is not None])
        self.assertEqual(load_folder_config(self.folder), config)

    def test_gui_initializes_configured_output_folder_and_preserves_legacy_paths(self):
        original = self.config_file.read_text(encoding="utf-8")
        for mode, output in (("custom", self.folder / "custom results"),
                             ("automatic", default_output_dir(self.folder)),
                             ("input", self.folder / "CV_postprocessing")):
            with self.subTest(mode=mode):
                self.config_file.write_text(original + f'--output-dir "{output.as_posix()}"\n', encoding="utf-8")
                config = load_folder_config(self.folder)
                app = SimpleNamespace(session=SimpleNamespace(base_config=config),
                                      output_mode=Mock(), custom_output=Mock(), output_changed=Mock())
                DropletApp._initialize_output(app)
                app.output_mode.set.assert_called_once_with(mode)
                app.custom_output.set.assert_called_once_with(str(output.resolve()))
                resolved = resolve_output_config(config, mode, str(output))
                self.assertEqual(resolved.output_csv, output / "cv_output.csv")
                self.assertEqual(resolved.output_dir, output)
        legacy = self.folder / "legacy" / "result.csv"
        debug = self.folder / "separate png"
        self.config_file.write_text(original + f'--output-csv "{legacy.as_posix()}"\n'
                                    f'--debug-dir "{debug.as_posix()}"\n', encoding="utf-8")
        config = load_folder_config(self.folder)
        app = SimpleNamespace(session=SimpleNamespace(base_config=config),
                              output_mode=Mock(), custom_output=Mock(), output_changed=Mock())
        DropletApp._initialize_output(app)
        app.output_mode.set.assert_called_once_with("custom")
        resolved = resolve_output_config(config, "custom", str(legacy.parent))
        self.assertEqual(resolved.output_csv, legacy)
        self.assertEqual(resolved.debug_dir, debug)

    def test_save_round_trip_preserves_auto_paths_flags_and_frame_range(self):
        config = replace(
            load_folder_config(self.folder), fps=1234.5, rim_to_rim=True,
            substrate="Al surface 名称", hydrophilicity="SHO", substrate_temp_deg_c=24.5,
            strong_threshold=None, weak_threshold=None, surface_y_px=None,
            surface_angle_deg=-1.25, surface_calibration_frame=None,
            droplet_size_ref_frame=3, start_frame=2, end_frame=5, top_exclusion_ratio=0.4,
            output_csv=self.folder / "output's folder" / "result.csv",
            debug_dir=self.folder / "output's folder" / "debug overlays",
        )
        save_folder_config(config)
        self.assertEqual(load_folder_config(self.folder), config)
        self.assertFalse((self.folder / "cv_config.txt.bak").exists())
        self.assertEqual(list(self.folder.glob(".cv_config-*.tmp")), [])

    def test_metadata_is_fixed_during_preview_and_does_not_affect_measurements(self):
        baseline = replace(load_folder_config(self.folder), debug_dir=None)
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write('--substrate "Al surface"\n--hydrophilicity SHO\n--substrate_temp_deg_c 24\n')
        session = PreviewSession.open(self.folder)
        session.preview(1, {"strong_threshold": 1100, "rim_to_rim": True})
        self.assertEqual(session.prepared.config.substrate, "Al surface")
        self.assertEqual(session.prepared.config.hydrophilicity, "SHO")
        self.assertEqual(session.prepared.config.substrate_temp_deg_c, 24.0)
        # Serialize actual batch measurements to compare NaNs and all CSV fields.
        from droplet_impact_cv.output import write_csv
        for rim_to_rim in (False, True):
            with self.subTest(rim_to_rim=rim_to_rim):
                expected = self.folder / "baseline.csv"
                actual = self.folder / "metadata.csv"
                write_csv(expected, analyze_sequence(replace(baseline, rim_to_rim=rim_to_rim)))
                write_csv(actual, analyze_sequence(replace(session.base_config, debug_dir=None, rim_to_rim=rim_to_rim)))
                self.assertEqual(expected.read_bytes(), actual.read_bytes())

    def test_save_backup_replaces_stale_backup_and_can_create_new_config(self):
        original = self.config_file.read_bytes()
        backup = self.folder / "cv_config.txt.bak"
        backup.write_text("stale", encoding="utf-8")
        config = replace(load_folder_config(self.folder), fps=2000, include_pre_impact=False)
        save_folder_config(config, backup=True)
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(load_folder_config(self.folder), config)
        self.config_file.unlink()
        save_folder_config(config, backup=True)
        self.assertEqual(load_folder_config(self.folder), config)
        self.assertEqual(backup.read_bytes(), original)

    def test_failed_save_restores_original_config(self):
        original = self.config_file.read_bytes()
        config = load_folder_config(self.folder)
        original_replace = Path.replace

        def fail_install(path, target):
            if path.suffix == ".tmp":
                raise PermissionError("Cannot install config")
            return original_replace(path, target)

        with patch.object(Path, "replace", fail_install):
            with self.assertRaises(PermissionError):
                save_folder_config(config, backup=True)
        self.assertEqual(self.config_file.read_bytes(), original)
        self.assertEqual(list(self.folder.glob(".cv_config-*.tmp")), [])

    def test_initial_preview_is_reference_and_discovery_stops_there(self):
        with patch("droplet_impact_cv.analysis.read_image", wraps=read_image) as reads:
            session = PreviewSession.open(self.folder)
        self.assertEqual(session.droplet_size_ref_frame, 4)
        self.assertEqual(session.initial_index, 3)
        self.assertEqual([call.args[0].name for call in reads.call_args_list], [
            "frame000001.tif", "frame000002.tif", "frame000003.tif", "frame000004.tif",
        ])

    def test_manual_recalibration_refreshes_reference_and_matches_batch(self):
        session = PreviewSession.open(self.folder)
        overrides = {"surface_y_px": 60, "surface_angle_deg": 0,
                     "strong_threshold": None, "weak_threshold": None,
                     "min_strong_threshold": 1000}
        session.preview(4, overrides, recalibrate=True)
        rows = analyze_sequence(replace(session.base_config, **overrides, debug_dir=None))
        self.assertEqual(session.droplet_size_ref_frame, rows[0].droplet_size_ref_frame)
        self.assertNotEqual(session.droplet_size_ref_frame, 4)
        expected = prepare_sequence(replace(session.base_config, **overrides))
        self.assertEqual(session.prepared.threshold, expected.threshold)
        self.assertEqual(session.prepared.weak_threshold, expected.weak_threshold)
        # Confirming the same line must also rerun automatic discovery.
        session.droplet_size_ref_frame = 999
        with patch("droplet_impact_cv.gui.prepare_sequence", wraps=prepare_sequence) as prepare:
            session.preview(4, overrides, recalibrate=True)
        prepare.assert_called_once()
        self.assertEqual(session.droplet_size_ref_frame, rows[0].droplet_size_ref_frame)

    def test_explicit_reference_bypasses_auto_clearance_in_preview_and_batch(self):
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write("--droplet-size-ref-frame 2\n")
        session = PreviewSession.open(self.folder)
        self.assertEqual(session.droplet_size_ref_frame, 2)
        self.assertEqual(session.initial_index, 1)
        rows = analyze_sequence(replace(session.base_config, debug_dir=None))
        self.assertTrue(all(row.droplet_size_ref_frame == 2 for row in rows))
        self.assertTrue(all(row.ref_clearance_px > row.ref_d_eq_volume_px / 2 for row in rows))

    def test_reference_override_freezes_selected_size_and_keeps_surface(self):
        config = replace(load_folder_config(self.folder), debug_dir=None)
        baseline = analyze_sequence(config)
        image = read_image(self.folder / "frame000005.tif")
        image[40:70, 30:90] = 2000
        tifffile.imwrite(self.folder / "frame000005.tif", image)
        selected = analyze_sequence(replace(config, droplet_size_ref_frame=5))
        self.assertTrue(all(row.droplet_size_ref_frame == 5 for row in selected))
        self.assertTrue(all(row.ref_d_eq_area_px > baseline[0].ref_d_eq_area_px for row in selected))
        self.assertTrue(all(row.surface_y_px == baseline[0].surface_y_px for row in selected))

    def test_explicit_reference_rejects_missing_out_of_range_or_empty_frames(self):
        config = replace(load_folder_config(self.folder), debug_dir=None)
        for overrides, message in (
            ({"droplet_size_ref_frame": 999}, "existing image number"),
            ({"droplet_size_ref_frame": 2, "start_frame": 3}, "existing image number"),
            ({"droplet_size_ref_frame": 1}, "measurable droplet"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_sequence(replace(config, **overrides))
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write("--droplet-size-ref-frame 1\n")
        with self.assertRaisesRegex(ValueError, "measurable droplet"):
            PreviewSession.open(self.folder)

    def test_frame_range_and_no_reference_fallback(self):
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write("--start-frame 2\n--end-frame 3\n")
        session = PreviewSession.open(self.folder)
        self.assertEqual([path.name for path in session.prepared.files], [
            "frame000002.tif", "frame000003.tif",
        ])
        self.assertIsNone(session.droplet_size_ref_frame)
        self.assertEqual(session.initial_index, 0)

    def test_frame_scrubbing_processes_only_selected_frame(self):
        session = PreviewSession.open(self.folder)
        with patch("droplet_impact_cv.analysis.read_image", wraps=read_image) as reads, \
                patch("droplet_impact_cv.gui.prepare_sequence", ) as prepare:
            before, after, _ = session.preview(4, {})
        self.assertEqual(before.shape, (120, 140))
        self.assertEqual(after.shape, (120, 140, 3))
        self.assertEqual([call.args[0].name for call in reads.call_args_list], ["frame000005.tif"])
        prepare.assert_not_called()

    def test_override_preview_matches_batch_and_never_writes_config_or_outputs(self):
        session = PreviewSession.open(self.folder)
        original = self.config_file.read_bytes()
        debug_dir = self.folder / "debug"
        debug_dir.mkdir()
        stale = debug_dir / "000099.png"
        stale.write_bytes(b"keep during preview")
        session.base_config = replace(session.base_config, debug_dir=debug_dir)
        _, baseline = session.prepared.measure(session.prepared.files[3])
        self.assertEqual(baseline.area_px2, 1200)
        overrides = {"strong_threshold": 2500.0}
        before, after, _ = session.preview(3, overrides)
        self.assertEqual(self.config_file.read_bytes(), original)
        self.assertEqual(stale.read_bytes(), b"keep during preview")
        self.assertEqual(list(debug_dir.iterdir()), [stale])
        self.assertFalse(np.array_equal(after[:, :, 0], before))  # Surface annotation remains.
        _, preview = session.prepared.measure(session.prepared.files[3])
        rows = analyze_sequence(replace(session.base_config, debug_dir=None, **overrides))
        self.assertEqual(preview.area_px2, 0)
        self.assertEqual(preview.area_px2, rows[3].droplet_area_px2)
        self.assertEqual(preview.d_spread_px, rows[3].d_spread_px)
        self.assertEqual(self.config_file.read_bytes(), original)

    def test_batch_progress_counts_working_frames(self):
        config = replace(load_folder_config(self.folder), debug_dir=None)
        events = []
        analyze_sequence(config, progress=lambda done, total: events.append((done, total)))
        self.assertEqual(events, [(number, 5) for number in range(6)])

    def test_experiment_overrides_change_units_time_and_calibration_in_batch(self):
        session = PreviewSession.open(self.folder)
        overrides = {"fps": 2000.0, "pixel_size_mm_per_px": 0.02, "surface_calibration_frame": 5}
        session.preview(3, overrides)
        self.assertEqual(session.prepared.config.surface_calibration_frame, 5)
        config = replace(session.base_config, debug_dir=None, **overrides)
        rows = analyze_sequence(config)
        self.assertEqual(rows[1].time_ms - rows[0].time_ms, 0.5)
        self.assertEqual(rows[0].pixel_size_mm_per_px, 0.02)
        self.assertEqual(rows[0].surface_calibration_frame, 5)
        self.assertAlmostEqual(rows[3].ref_d_eq_area_mm, rows[3].ref_d_eq_area_px * 0.02)


class GuiControlTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.root.update_idletasks()
        # Release Tk callback cycles on the UI thread before workflow workers start.
        self.addCleanup(gc.collect)
        self.addCleanup(self.root.destroy)

    def test_save_dialog_cancel_backup_and_overwrite(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cv_config.txt"
            path.write_text("--fps 4000\n", encoding="utf-8")
            original = path.read_bytes()
            config = replace(load_folder_config(Path(folder)), fps=2000)
            # Exercise the GUI command without loading an image sequence.
            app = SimpleNamespace(
                session=SimpleNamespace(), processing=False, selecting_surface=False,
                root=self.root, batch_config=lambda: config, set_status=lambda *args: None,
            )
            with patch("droplet_impact_cv.gui.messagebox.askyesnocancel", return_value=None):
                DropletApp.save_parameters(app)
            self.assertEqual(path.read_bytes(), original)
            with patch("droplet_impact_cv.gui.messagebox.askyesnocancel", return_value=True):
                DropletApp.save_parameters(app)
            backup = path.with_name("cv_config.txt.bak")
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(load_folder_config(Path(folder)), config)
            app.batch_config = lambda: replace(config, fps=3000)
            with patch("droplet_impact_cv.gui.messagebox.askyesnocancel", return_value=False):
                DropletApp.save_parameters(app)
            self.assertEqual(load_folder_config(Path(folder)).fps, 3000)
            self.assertEqual(backup.read_bytes(), original)

    def test_disabled_slider_tracks_loaded_auto_values_and_reset(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "strong_threshold")
        control = ParameterControl(self.root, spec, lambda: None)
        control.slider.configure(state="disabled")
        control.set_value(None, 1500.0)
        self.assertEqual(control.slider.get(), 1500.0)
        self.assertEqual(str(control.slider["state"]), "disabled")
        control.set_value(None, 2200.0, remember=False)
        self.assertEqual(control.slider.get(), 2200.0)
        control.reset_value()
        self.assertEqual(control.slider.get(), 1500.0)
        control.set_value(2000.0, 2000.0, remember=False)
        self.assertEqual(control.slider.get(), 2000.0)
        self.assertEqual(str(control.slider["state"]), "normal")

    def test_initial_values_preserve_precision_and_auto(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "strong_threshold")
        control = ParameterControl(self.root, spec, lambda: None)
        value = 1234.56789012345
        control.set_value(value, value)
        self.assertEqual(control.get_value(), value)
        control.set_value(None, value)
        self.assertIsNone(control.get_value())
        self.assertEqual(str(control.entry["state"]), "disabled")

    def test_entry_updates_slider_and_invalid_values_block_processing(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "weak_reach_px")
        control = ParameterControl(self.root, spec, lambda: None)
        control.set_value(40, 40)
        control.value.set("500")
        self.assertEqual(control.slider.get(), 500)
        self.assertEqual(control.get_value(), 500)
        control.value.set("2.5")
        self.assertFalse(control.valid)
        with self.assertRaises(ValueError):
            control.get_value()

    def test_slider_track_click_jumps_to_position_and_preserves_thumb_drag(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "weak_reach_px")
        changes = []
        control = ParameterControl(self.root, spec, lambda: changes.append(control.get_value()))
        control.pack(fill="x")
        self.root.geometry("600x250")
        self.root.deiconify()
        self.root.update_idletasks()
        # Include values outside the suggested range, as when loading a config.
        control.set_value(500, 500)
        width, y = control.slider.winfo_width(), control.slider.winfo_height() // 2
        for fraction in (0.25, 0.75, 0, 1):
            x = round((width - 1) * fraction)
            expected = round(float(control.slider.get(x, y)))
            control.slider.event_generate("<ButtonPress-1>", x=x, y=y)
            control.slider.event_generate("<ButtonRelease-1>", x=x, y=y)
            self.assertEqual(control.get_value(), expected)
            self.assertEqual(control.slider.get(), expected)
        self.assertTrue(changes)
        control.set_value(100, 100)
        self.root.update_idletasks()
        thumb = [x for x in range(width) if control.slider.identify(x, y).endswith("slider")]
        self.assertTrue(thumb)
        x = thumb[len(thumb) // 2]
        control.slider.event_generate("<ButtonPress-1>", x=x, y=y)
        control.slider.event_generate("<B1-Motion>", x=x + width // 4, y=y)
        control.slider.event_generate("<ButtonRelease-1>", x=x + width // 4, y=y)
        self.assertGreater(control.get_value(), 100)
        self.assertLess(control.get_value(), spec.high)

    def test_slider_track_click_ignores_auto_and_disabled_controls(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "strong_threshold")
        changes = []
        control = ParameterControl(self.root, spec, lambda: changes.append(True))
        control.pack(fill="x")
        self.root.deiconify()
        self.root.update_idletasks()
        control.set_value(None, 1500)
        control.slider.event_generate("<ButtonPress-1>", x=10, y=5)
        self.assertIsNone(control.get_value())
        self.assertEqual(control.slider.get(), 1500)
        control.set_value(2000, 2000)
        control.slider.configure(state="disabled")
        control.slider.event_generate("<ButtonPress-1>", x=10, y=5)
        self.assertEqual(control.get_value(), 2000)
        self.assertFalse(changes)

    def test_ratio_accepts_small_positive_values_and_rejects_nonfinite(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "weak_threshold_ratio")
        control = ParameterControl(self.root, spec, lambda: None)
        control.set_value(0.00001, 0.00001)
        self.assertEqual(control.get_value(), 0.00001)
        for value in ("0", "1.1", "nan", "inf"):
            control.value.set(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.get_value()

    def test_top_exclusion_control_accepts_endpoints_and_rejects_invalid_values(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "top_exclusion_ratio")
        control = ParameterControl(self.root, spec, lambda: None)
        for value in (0, 0.25, 0.5, 1):
            control.set_value(value, value)
            self.assertEqual(control.get_value(), value)
        for value in ("-0.1", "1.1", "nan", "inf"):
            control.value.set(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.get_value()

    def test_individual_reset_restores_value_and_auto_without_changing_baseline(self):
        changed = []
        spec = next(spec for spec in PARAMETERS if spec.name == "strong_threshold")
        control = ParameterControl(self.root, spec, lambda: changed.append(True))
        control.set_value(None, 1500.0)
        control.auto.set(False)
        control.value.set("2000")
        control.reset_button.invoke()
        self.assertIsNone(control.get_value())
        self.assertTrue(control.auto.get())
        self.assertEqual(control.value.get(), "1500.0")
        control.set_value(None, 2200.0, remember=False)
        control.reset_button.invoke()
        self.assertEqual(control.value.get(), "1500.0")
        self.assertTrue(changed)

    def test_experiment_parameters_validate_positive_values_and_frame_integers(self):
        for name in ("fps", "pixel_size_mm_per_px", "surface_calibration_frame"):
            spec = next(spec for spec in PARAMETERS if spec.name == name)
            control = ParameterControl(self.root, spec, lambda: None)
            control.set_value(1, 1)
            control.value.set("-1")
            with self.subTest(name=name), self.assertRaises(ValueError):
                control.get_value()
        control.value.set("1.5")
        with self.assertRaises(ValueError):
            control.get_value()


class GuiWorkflowTests(unittest.TestCase):
    def test_combobox_wheel_does_not_change_selection(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = DropletApp(root)
        self.addCleanup(app.close)
        combo = app.reflection_combo
        combo.configure(state="readonly")
        combo.current(1)
        root.update()
        with patch.object(app, "schedule_preview") as preview:
            for sequence, options in (
                ("<MouseWheel>", {"delta": 120}),
                ("<MouseWheel>", {"delta": -120}),
                ("<Button-4>", {}),
                ("<Button-5>", {}),
            ):
                combo.event_generate(sequence, **options)
                root.update()
                self.assertEqual(combo.current(), 1)
            preview.assert_not_called()
            combo.current(2)
            combo.event_generate("<<ComboboxSelected>>")
            root.update()
            self.assertEqual(app.reflection.get(), "none")
            preview.assert_called_once()

    def test_manual_surface_selection_zoom_frame_switch_and_auto_preservation(self):
        PreviewTests.setUp(self)
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = DropletApp(root)
        self.addCleanup(app.close)

        def pump():
            deadline = time.monotonic() + 10
            while (not app.preview_ready if not app.selecting_surface else
                   app.pending is not None or app.preview_future is None or not app.preview_future.done()
                   or app.displayed_frame_index != round(app.frame_slider.get())):
                root.update()
                if time.monotonic() > deadline:
                    self.fail(app.status.get())
                time.sleep(0.01)
            root.update()

        def click(x, y):
            canvas = app.preview_labels[0]
            scale, left, top = app.viewports[0].transform(
                app.images[0].shape, max(1, canvas.winfo_width()), max(1, canvas.winfo_height()),
            )
            app.select_surface_point(SimpleNamespace(x=left + x * scale, y=top + y * scale))

        with patch("droplet_impact_cv.gui.filedialog.askdirectory", return_value=str(self.folder)):
            app.choose_folder()
        pump()
        app.start_surface_selection()
        pump()
        click(20, 70)
        app.navigate_frame("last")
        self.assertEqual(app.surface_points, [])
        pump()
        app.controls["surface_angle_deg"].set_value(None, 0, remember=False)
        app.controls["surface_calibration_frame"].set_value(4, 4, remember=False)
        app.viewports[0].zoom_at(0.5, 0.5, 2, app.images[0].shape, 1, 1)
        click(20, 70)
        click(100, 78)
        self.assertEqual(len(app.preview_labels[0].find_all()), 4)  # Image, line, two points.
        self.assertEqual(str(app.process["state"]), "disabled")
        with patch("droplet_impact_cv.gui.messagebox.askyesno", return_value=True) as prompt:
            app.confirm_surface_selection()
        prompt.assert_called_once()
        self.assertEqual(app.controls["surface_calibration_frame"].get_value(), 5)
        self.assertEqual(app.controls["surface_y_px"].get_value(), 75)
        self.assertAlmostEqual(float(app.controls["surface_angle_deg"].value.get()), np.degrees(np.arctan(0.1)))
        self.assertTrue(app.controls["surface_angle_deg"].auto.get())
        self.assertFalse(app.controls["surface_y_px"].auto.get())
        self.assertFalse(app.controls["surface_calibration_frame"].auto.get())
        self.assertFalse(app.selecting_surface)
        pump()
        self.assertEqual(float(app.controls["surface_angle_deg"].value.get()), app.session.prepared.surface_line.angle_deg)
        app.reset_image_size(0)
        self.assertEqual(app.viewports[0], PreviewViewport())

        # A validation error clears preview_ready, but raw frames remain selectable.
        app.controls["surface_calibration_frame"].set_value(999, 999, remember=False)
        app.controls["surface_y_px"].set_value(None, 75, remember=False)
        app.schedule_preview()
        self.assertFalse(app.preview_ready)
        self.assertTrue(app.status.get().startswith("Error:"))
        app.start_surface_selection()
        app.navigate_frame(-1)
        pump()
        click(20, 70)
        click(100, 78)
        self.assertEqual(len(app.surface_points), 2)
        self.assertEqual(str(app.surface_confirm["state"]), "normal")
        with patch("droplet_impact_cv.gui.messagebox.askyesno", return_value=False) as prompt:
            app.confirm_surface_selection()
        prompt.assert_called_once()
        self.assertEqual(app.controls["surface_calibration_frame"].get_value(), 4)
        self.assertEqual(app.controls["surface_y_px"].get_value(), 75)
        self.assertAlmostEqual(app.controls["surface_angle_deg"].get_value(), np.degrees(np.arctan(0.1)))
        for name in ("surface_calibration_frame", "surface_y_px", "surface_angle_deg"):
            self.assertFalse(app.controls[name].auto.get())
        pump()
        self.assertEqual(app.controls["droplet_size_ref_frame"].get_value(), None)
        self.assertEqual(int(app.controls["droplet_size_ref_frame"].value.get()), app.session.droplet_size_ref_frame)

    def test_load_preview_scrub_edit_reset_and_process(self):
        PreviewTests.setUp(self)
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = DropletApp(root)
        self.addCleanup(app.close)
        output = self.folder / "result.csv"
        debug = self.folder / "debug"
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write(f'--output-csv "{output.as_posix()}"\n--debug-dir "{debug.as_posix()}"\n')
        original = self.config_file.read_bytes()

        def pump(condition):
            deadline = time.monotonic() + 10
            while not condition():
                root.update()
                if time.monotonic() > deadline:
                    self.fail(app.status.get())
                time.sleep(0.01)
            root.update()

        with patch("droplet_impact_cv.gui.filedialog.askdirectory", return_value=str(self.folder)), \
                patch("droplet_impact_cv.gui.messagebox.showerror", side_effect=AssertionError):
            app.choose_folder()
            pump(lambda: app.status.get().startswith("Ready."))
            self.assertEqual(app.session.droplet_size_ref_frame, 4)
            reference_control = app.controls["droplet_size_ref_frame"]
            self.assertTrue(reference_control.auto.get())
            self.assertEqual(reference_control.value.get(), "4")
            reference_control.set_value(2, 2, remember=False)
            app.schedule_preview()
            pump(lambda: app.preview_ready)
            self.assertEqual(app.batch_config().droplet_size_ref_frame, 2)
            reference_control.value.set("999")
            self.assertEqual(str(app.process["state"]), "disabled")
            self.assertIn("Droplet size reference frame", app.status.get())
            reference_control.reset_button.invoke()
            pump(lambda: app.preview_ready)
            self.assertIsNone(app.batch_config().droplet_size_ref_frame)
            self.assertEqual(len(app.photos), 2)
            self.assertEqual(app.output_mode.get(), "custom")
            self.assertEqual(app.batch_config().output_dir, self.folder)
            self.assertEqual(app.batch_config().output_csv, output)
            self.assertEqual(app.batch_config().debug_dir, debug)
            app.output_mode.set("automatic")
            app.output_changed()
            self.assertEqual(app.batch_config().output_csv.parent, default_output_dir(self.folder))
            app.output_mode.set("custom")
            app.custom_output.set("")
            self.assertEqual(str(app.process["state"]), "disabled")
            selected = self.folder / "manual-output"
            with patch("droplet_impact_cv.gui.filedialog.askdirectory", return_value=str(selected)):
                app.choose_output_folder()
            self.assertEqual(app.batch_config().debug_dir, selected / "debug_overlays")
            app.output_mode.set("input")
            app.output_changed()
            self.assertEqual(app.batch_config().output_csv.parent, self.folder / "CV_postprocessing")
            app.frame_slider.set(4)
            pump(lambda: "frame000005.tif" in app.details.get() and app.status.get().startswith("Ready."))
            app.controls["weak_reach_px"].value.set("bad")
            self.assertEqual(str(app.process["state"]), "disabled")
            app.controls["weak_reach_px"].value.set("50")
            pump(lambda: app.status.get().startswith("Ready."))
            self.assertEqual(app.overrides()["weak_reach_px"], 50)
            app.reset_parameters()
            pump(lambda: app.status.get().startswith("Ready."))
            self.assertEqual(app.overrides()["weak_reach_px"], 40)
            app.process_all()
            pump(lambda: app.status.get().startswith("Complete."))
        selected_output = self.folder / "CV_postprocessing"
        self.assertTrue((selected_output / "cv_output.csv").exists())
        self.assertTrue((selected_output / "run_params.json").is_file())
        self.assertEqual(len(list((selected_output / "debug_overlays").glob("*.png"))), 5)
        self.assertFalse(output.exists())
        self.assertFalse(debug.exists())
        self.assertEqual(self.config_file.read_bytes(), original)

    def test_group_reset_playback_and_error_feedback(self):
        PreviewTests.setUp(self)
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = DropletApp(root)
        self.addCleanup(app.close)

        def pump(condition):
            deadline = time.monotonic() + 10
            while not condition():
                root.update()
                if time.monotonic() > deadline:
                    self.fail(app.status.get())
                time.sleep(0.005)
            root.update()

        with patch("droplet_impact_cv.gui.filedialog.askdirectory", return_value=str(self.folder)), \
                patch("droplet_impact_cv.gui.messagebox.showerror") as popup:
            app.choose_folder()
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["busy"])
            pump(lambda: app.preview_ready)
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["ready"])
            self.assertEqual(app.playback_fps.get(), "5")
            self.assertEqual(app.controls["fps"].get_value(), 4000.0)
            for name, control in app.controls.items():
                with self.subTest(parameter=name):
                    self.assertAlmostEqual(control.slider.get(), float(control.value.get()))
            group = app.groups["Experiment and scale"]
            group.header.invoke()
            self.assertFalse(group.expanded)
            group.header.invoke()
            self.assertTrue(group.expanded)
            app.controls["fps"].value.set("2000")
            app.controls["weak_reach_px"].value.set("60")
            app.controls["fps"].reset_button.invoke()
            self.assertEqual(app.controls["fps"].get_value(), 4000.0)
            self.assertEqual(app.controls["weak_reach_px"].get_value(), 60)
            app.rim_to_rim.set(True)
            app.reset_choice("rim_to_rim")
            self.assertFalse(app.rim_to_rim.get())
            app.reflection.set("none")
            app.reset_choice("reflection_mode")
            self.assertEqual(app.reflection.get(), "auto")
            pump(lambda: app.preview_ready)
            app.navigate_frame("first")
            pump(lambda: app.preview_ready and round(app.frame_slider.get()) == 0)
            app.playback_fps.set("40")
            app.play(1)
            pump(lambda: app.play_direction == 0 and round(app.frame_slider.get()) == 4)
            app.play(-1)
            pump(lambda: app.play_direction == 0 and round(app.frame_slider.get()) == 0)
            app.navigate_frame("last")
            pump(lambda: app.preview_ready and round(app.frame_slider.get()) == 4)
            app.navigate_frame(-1)
            pump(lambda: app.preview_ready and round(app.frame_slider.get()) == 3)
            app.navigate_frame(1)
            pump(lambda: app.preview_ready and round(app.frame_slider.get()) == 4)
            app.play(-1)
            app.pause()
            pump(lambda: app.preview_ready)
            paused_index = round(app.frame_slider.get())
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                root.update()
                time.sleep(0.005)
            self.assertEqual(round(app.frame_slider.get()), paused_index)
            app.playback_fps.set("0")
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["error"])
            self.assertEqual(app.play_direction, 0)
            popup.assert_not_called()
            app.speed_reset.invoke()
            self.assertEqual(app.playback_fps.get(), "5")
            app.controls["surface_calibration_frame"].auto.set(False)
            app.controls["surface_calibration_frame"].value.set("999")
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["error"])
            self.assertEqual(str(app.process["state"]), "disabled")
            popup.assert_not_called()
            app.controls["surface_calibration_frame"].reset_button.invoke()
            pump(lambda: app.preview_ready)
            # Recoverable preview-setting errors are red and never show a dialog.
            def preview_error():
                raise ValueError("Invalid preview setting")
            app._submit("preview", app.generation, preview_error)
            root.update()
            pump(lambda: app.status.get().startswith("Error:"))
            popup.assert_not_called()
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["error"])
            # Critical IO/batch errors retain the existing modal dialog.
            def batch_error():
                raise PermissionError("Cannot write results")
            app._submit("processed", app.generation, batch_error)
            pump(lambda: popup.called)
            popup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
