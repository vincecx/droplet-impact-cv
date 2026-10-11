from __future__ import annotations

import sys
import time
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

from droplet_impact_cv.analysis import analyze_sequence
from droplet_impact_cv.gui import (
    PARAMETERS, STATUS_COLORS, DropletApp, ParameterControl, PreviewSession, PreviewViewport,
    load_folder_config, resolve_output_config, surface_values_from_points,
)
from droplet_impact_cv.cli import default_output_dir
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
        self.folder = Path(self.temporary.name)
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
                self.assertEqual(result.output_csv, folder / "configured.csv")
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

    def test_initial_preview_is_reference_and_discovery_stops_there(self):
        with patch("droplet_impact_cv.analysis.read_image", wraps=read_image) as reads:
            session = PreviewSession.open(self.folder)
        self.assertEqual(session.ref_frame, 4)
        self.assertEqual(session.initial_index, 3)
        self.assertEqual([call.args[0].name for call in reads.call_args_list], [
            "frame000001.tif", "frame000002.tif", "frame000003.tif", "frame000004.tif",
        ])

    def test_frame_range_and_no_reference_fallback(self):
        with self.config_file.open("a", encoding="utf-8") as handle:
            handle.write("--start-frame 2\n--end-frame 3\n")
        session = PreviewSession.open(self.folder)
        self.assertEqual([path.name for path in session.prepared.files], [
            "frame000002.tif", "frame000003.tif",
        ])
        self.assertIsNone(session.ref_frame)
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
        overrides = {"fps": 2000.0, "pixel_size_mm_per_px": 0.02, "calibration_frame": 5}
        session.preview(3, overrides)
        self.assertEqual(session.prepared.config.calibration_frame, 5)
        config = replace(session.base_config, debug_dir=None, **overrides)
        rows = analyze_sequence(config)
        self.assertEqual(rows[1].time_ms - rows[0].time_ms, 0.5)
        self.assertEqual(rows[0].pixel_size_mm_per_px, 0.02)
        self.assertEqual(rows[0].calibration_frame, 5)
        self.assertAlmostEqual(rows[3].ref_d_eq_area_mm, rows[3].ref_d_eq_area_px * 0.02)


class GuiControlTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.root.update_idletasks()
        self.addCleanup(self.root.destroy)

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

    def test_ratio_accepts_small_positive_values_and_rejects_nonfinite(self):
        spec = next(spec for spec in PARAMETERS if spec.name == "weak_threshold_ratio")
        control = ParameterControl(self.root, spec, lambda: None)
        control.set_value(0.00001, 0.00001)
        self.assertEqual(control.get_value(), 0.00001)
        for value in ("0", "1.1", "nan", "inf"):
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
        for name in ("fps", "pixel_size_mm_per_px", "calibration_frame"):
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
            while not app.preview_ready:
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
        click(20, 70)
        app.navigate_frame("last")
        self.assertEqual(app.surface_points, [])
        pump()
        app.controls["surface_angle_deg"].set_value(None, 0, remember=False)
        app.controls["calibration_frame"].set_value(4, 4, remember=False)
        app.viewports[0].zoom_at(0.5, 0.5, 2, app.images[0].shape, 1, 1)
        click(20, 70)
        click(100, 78)
        self.assertEqual(len(app.preview_labels[0].find_all()), 4)  # Image, line, two points.
        self.assertEqual(str(app.process["state"]), "disabled")
        app.confirm_surface_selection()
        self.assertEqual(app.controls["calibration_frame"].get_value(), 5)
        self.assertEqual(app.controls["surface_y_px"].get_value(), 75)
        self.assertAlmostEqual(float(app.controls["surface_angle_deg"].value.get()), np.degrees(np.arctan(0.1)))
        self.assertTrue(app.controls["surface_angle_deg"].auto.get())
        self.assertFalse(app.controls["surface_y_px"].auto.get())
        self.assertFalse(app.controls["calibration_frame"].auto.get())
        self.assertFalse(app.selecting_surface)
        pump()
        self.assertEqual(float(app.controls["surface_angle_deg"].value.get()), app.session.prepared.surface_line.angle_deg)
        app.reset_image_size(0)
        self.assertEqual(app.viewports[0], PreviewViewport())

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
            self.assertEqual(app.session.ref_frame, 4)
            self.assertEqual(len(app.photos), 2)
            self.assertEqual(app.output_mode.get(), "input")
            self.assertEqual(app.batch_config().debug_dir, self.folder / "CV_postprocessing" / "debug_overlays")
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
        self.assertTrue((selected_output / "result.csv").exists())
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
            app.hydrophobic.set(True)
            app.reset_choice("hydrophobic")
            self.assertFalse(app.hydrophobic.get())
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
            app.controls["calibration_frame"].auto.set(False)
            app.controls["calibration_frame"].value.set("999")
            self.assertEqual(str(app.status_label["foreground"]), STATUS_COLORS["error"])
            self.assertEqual(str(app.process["state"]), "disabled")
            popup.assert_not_called()
            app.controls["calibration_frame"].reset_button.invoke()
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
