from __future__ import annotations

import argparse
import math
import queue
import shlex
import tempfile
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk
from typing import Callable

import cv2
import numpy as np

from .analysis import (
    PreparedSequence, analyze_sequence, is_reference_measurement, prepare_sequence,
)
from .cli import CONFIG_FILENAME, build_parser, config_from_args, default_output_dir, parse_args
from .imaging import frame_number_from_filename, read_image
from .models import CONFIG_PARAMETER_ORDER, OUTPUT_CSV_FILENAME, AnalysisConfig
from .output import RUN_PARAMS_FILENAME, write_csv, write_run_params
from .visualization import render_debug_overlay


def load_folder_config(folder: Path) -> AnalysisConfig:
    """Load only this folder's config and defaults, never process CLI overrides."""
    # argparse normally exits on invalid options. Convert that into a GUI error.
    from contextlib import redirect_stderr
    from io import StringIO

    errors = StringIO()
    try:
        with redirect_stderr(errors):
            return config_from_args(parse_args([str(folder)]))
    except SystemExit as error:
        raise ValueError(errors.getvalue().strip()) from error


def save_folder_config(config: AnalysisConfig, backup: bool = False) -> Path:
    """Write reloadable CLI settings, preserving Auto by omitting None values."""
    lines = ["# Droplet Impact CV analysis settings"]
    actions = {action.dest: action for action in build_parser()._actions if action.option_strings}
    for name in CONFIG_PARAMETER_ORDER:
        action = actions.get(name)
        if action is None:
            continue
        value = getattr(config, action.dest)
        if action.dest == "output_dir" and value is None:
            value = config.output_csv.parent
        if value is None:
            continue
        option = next(option for option in action.option_strings if option.startswith("--"))
        if isinstance(action, argparse.BooleanOptionalAction):
            lines.append(option if value else "--no-" + option[2:])
        else:
            if isinstance(value, Path):
                value = value.resolve().as_posix()
            elif isinstance(value, bool):
                value = int(value)
            lines.append(f"{option} {shlex.quote(str(value))}")
    path = config.input_dir / CONFIG_FILENAME
    backup_path = path.with_name(path.name + ".bak")
    # Finish writing before moving the original or replacing the destination.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=config.input_dir,
                                     prefix=".cv_config-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write("\n".join(lines) + "\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    moved = False
    try:
        if backup and path.exists():
            path.replace(backup_path)
            moved = True
        temporary.replace(path)
    except OSError:
        if moved:
            backup_path.replace(path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return path


def resolve_output_config(
    config: AnalysisConfig, mode: str, custom_folder: str = "",
) -> AnalysisConfig:
    if mode == "automatic":
        folder = default_output_dir(config.input_dir)
    elif mode == "input":
        folder = config.input_dir.resolve() / "CV_postprocessing"
    elif mode == "custom":
        if not custom_folder.strip():
            raise ValueError("Choose or enter an output folder.")
        folder = Path(custom_folder.strip()).expanduser().resolve()
    else:
        raise ValueError("Select a valid output location.")
    if folder.exists() and not folder.is_dir():
        raise ValueError("The output folder must be a directory.")
    if mode == "custom" and folder == (config.output_dir or config.output_csv.parent).resolve():
        # Preserve legacy per-file overrides when opening a configured folder.
        return replace(config, output_dir=folder)
    return replace(
        config, output_dir=folder, output_csv=folder / OUTPUT_CSV_FILENAME,
        debug_dir=folder / "debug_overlays",
    )


@dataclass
class PreviewSession:
    base_config: AnalysisConfig
    prepared: PreparedSequence
    initial_index: int
    droplet_size_ref_frame: int | None
    applied_config: AnalysisConfig | None = None

    @classmethod
    def open(
        cls, folder: Path, report: Callable[[str], None] = lambda _: None,
        stopped: Callable[[], bool] = lambda: False,
    ) -> PreviewSession:
        config = load_folder_config(folder)
        prepared = prepare_sequence(config)
        if config.droplet_size_ref_frame is not None:
            index = next(index for index, path in enumerate(prepared.files)
                         if frame_number_from_filename(path) == config.droplet_size_ref_frame)
            if stopped():
                raise RuntimeError("Operation cancelled.")
            _, measurement = prepared.measure(prepared.files[index])
            if measurement.d_eq_volume_px is None or measurement.d_eq_area_px is None:
                raise ValueError("Droplet size reference frame must contain a measurable droplet.")
            return cls(config, prepared, index, config.droplet_size_ref_frame)
        # Reference discovery happens once at open, stopping at the first match.
        # It never writes CSVs or removes/writes diagnostic files.
        for index, path in enumerate(prepared.files):
            if stopped():
                raise RuntimeError("Operation cancelled.")
            report(f"Finding droplet size reference frame: {index + 1} / {len(prepared.files)}")
            _, measurement = prepared.measure(path)
            if is_reference_measurement(measurement):
                return cls(config, prepared, index, frame_number_from_filename(path))
        return cls(config, prepared, 0, None)

    def raw_preview(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        path = self.prepared.files[index]
        image = read_image(path)
        lo, hi = np.percentile(image, [0.5, 99.5])
        before = np.clip((image - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8)
        return before, cv2.cvtColor(before, cv2.COLOR_GRAY2RGB), path.name

    def preview(self, index: int, overrides: dict, recalibrate: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
        config = replace(self.base_config, **overrides)
        if recalibrate or config != (self.applied_config or self.base_config):
            # Reuse the fixed background and file range; calibration and sampled
            # automatic thresholds follow the same rules as full analysis.
            self.prepared = prepare_sequence(
                config, self.prepared.files, self.prepared.background,
            )
            self.applied_config = config
        if recalibrate:
            reference = config.droplet_size_ref_frame
            if reference is None:
                for path in self.prepared.files:
                    _, candidate = self.prepared.measure(path)
                    if is_reference_measurement(candidate):
                        reference = frame_number_from_filename(path)
                        break
            self.droplet_size_ref_frame = reference
        path = self.prepared.files[index]
        image, measurement = self.prepared.measure(path)
        scale = config.pixel_size_mm_per_px
        after = render_debug_overlay(
            image, measurement.mask, self.prepared.surface_line,
            measurement.contact_bbox, frame_number_from_filename(path),
            measurement.d_spread_px * scale, measurement.rim_bbox,
            None if measurement.d_contact_px is None else measurement.d_contact_px * scale,
            None if measurement.d_rim_px is None else measurement.d_rim_px * scale,
        )
        lo, hi = np.percentile(image, [0.5, 99.5])
        before = np.clip((image - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8)
        details = (
            f"{path.name} | Area: {measurement.area_px2} px2 | "
            f"Strong threshold: {self.prepared.threshold:g} | "
            f"Weak threshold: {self.prepared.weak_threshold:g}"
        )
        if self.prepared.threshold_sampling_method != "manual":
            details += (
                f" | Auto samples: {self.prepared.threshold_sampling_method}"
                f" (anchor: {self.prepared.threshold_anchor_frame or 'none'})"
            )
        if not self.prepared.surface_is_calibrated:
            details += " | Surface uncalibrated: check the line or select two surface points."
        if measurement.shadow_removed_px2:
            details += f" | Shadow fringe removed: {measurement.shadow_removed_px2} px2"
        if measurement.segmentation_warning:
            details += " | " + measurement.segmentation_warning
        return before, cv2.cvtColor(after, cv2.COLOR_BGR2RGB), details


@dataclass(frozen=True)
class Parameter:
    name: str
    label: str
    low: float
    high: float
    step: float
    description: str
    automatic: bool = False


PARAMETERS = (
    Parameter("fps", "Camera rate (fps)", 1, 20000, 1,
              "Camera recording rate, not preview speed."),
    Parameter("pixel_size_mm_per_px", "Image scale (mm/px)", 0.000001, 0.1, 0.000001,
              "Length represented by one image pixel, from your scale calibration."),
    Parameter("droplet_size_ref_frame", "Droplet size reference frame", 0, 10000, 1,
              "Image number used for both reference equivalent diameters. Choose a frame with a clearly visible drop before impact. Auto selects the first frame whose substrate clearance is at most half its volume-equivalent diameter.", True),
    Parameter("surface_calibration_frame", "Surface calibration frame", 0, 10000, 1,
              "Choose an image number after impact where the drop meets the substrate. This image sets the substrate line; changing it can move that line. Auto uses the clean background.", True),
    Parameter("top_exclusion_ratio", "Exclude top image fraction", 0, 1, 0.01,
              "Fraction of image height excluded from droplet detection at the top. 0 includes all rows; 1 excludes all rows. Background and surface calibration use the full image."),
    Parameter("strong_threshold", "Main drop threshold", 1, 4095, 1,
              "How different a pixel must be from the clean background to count as liquid. Lower values detect more of the drop, but may also include image noise.", True),
    Parameter("min_strong_threshold", "Min auto threshold", 1, 4095, 1,
              "Lowest main threshold allowed in Auto mode (default 200). Auto samples around contact or a sustained shape change, with background noise setting the effective threshold."),
    Parameter("weak_threshold", "Faint liquid threshold", 1, 4095, 1,
              "Allows faint pixels next to clearly detected liquid to join the drop. Lower values can recover transparent edges, but may make the outline too large.", True),
    Parameter("weak_threshold_ratio", "Faint / main ratio", 0.001, 1, 0.001,
              "Sets the faint threshold as a fraction of the main threshold when faint detection is Auto. Lower ratios include fainter liquid near the drop."),
    Parameter("weak_reach_px", "Faint liquid reach (px)", 0, 200, 1,
              "How far faint liquid can extend from clearly detected liquid. 0 disables faint extension. Increase to recover a thin sheet or rim; too much may include unrelated image changes."),
    Parameter("min_droplet_area_px2", "Min drop area (px2)", 1, 10000, 1,
              "Ignore detected objects smaller than this area. Increase to remove small specks; too high may discard small drops."),
    Parameter("morphology_radius_px", "Outline cleanup size (px)", 0, 15, 1,
              "Size of the image cleanup tool. Increase to close small gaps and remove specks; too much can smooth away real fine details."),
    Parameter("surface_band_px", "Contact band (px)", 0, 30, 1,
              "Distance above and below the substrate line used to find contact width. Increase if the contact edge is blurred; too much may count liquid away from the surface."),
    Parameter("foreground_depth_px", "Search depth below surface (px)", 0, 500, 1,
              "How far below the substrate line to look for image changes. Increase if the contact region is cut off; too much may include reflections."),
    Parameter("min_contact_px", "Min visible drop pixels (px)", 0, 1000, 1,
              "Minimum number of detected pixels above the substrate for an object to be considered a drop. Increase to reject tiny specks; too high may lose small drops."),
    Parameter("surface_y_px", "Surface center y (px)", 0, 2048, 1,
              "Position (y-coordinate) of the substrate line at the image center. Larger values move it downward. Auto uses the detected line.", True),
    Parameter("surface_angle_deg", "Surface tilt (deg)", -45, 45, 0.1,
              "Tilt of the substrate line. Positive values rotate it clockwise. Adjust to match the visible substrate; Auto uses the detected tilt.", True),
    Parameter("mirror_min_depth_px", "Min mirror depth (px)", 0, 500, 1,
              "Minimum distance a mirror calibration candidate must extend below the coarse substrate line. Lower values allow shallower reflections; Auto reflection mode also uses this check."),
)

PARAMETER_GROUPS = (
    ("Experiment and scale", ("fps", "pixel_size_mm_per_px")),
    ("Substrate and width measurement", ("droplet_size_ref_frame", "surface_calibration_frame", "surface_y_px", "surface_angle_deg", "surface_band_px", "mirror_min_depth_px")),
    ("Liquid detection", ("top_exclusion_ratio", "strong_threshold", "min_strong_threshold", "weak_threshold", "weak_threshold_ratio", "weak_reach_px", "foreground_depth_px")),
    ("Drop selection and outline cleanup", ("min_droplet_area_px2", "min_contact_px", "morphology_radius_px")),
)
STATUS_COLORS = {"busy": "#a66b00", "ready": "#16803c", "error": "#b42318", "idle": "#526071"}


@dataclass
class PreviewViewport:
    zoom: float = 1.0
    # Visible center in normalized image coordinates, retained across frames.
    center_x: float = 0.5
    center_y: float = 0.5

    def transform(self, shape, width, height):
        image_height, image_width = shape[:2]
        scale = min(width / image_width, height / image_height) * self.zoom
        return scale, width / 2 - self.center_x * image_width * scale, height / 2 - self.center_y * image_height * scale

    def image_point(self, x, y, shape, width, height):
        scale, left, top = self.transform(shape, width, height)
        ix, iy = (x - left) / scale, (y - top) / scale
        if 0 <= ix <= shape[1] - 1 and 0 <= iy <= shape[0] - 1:
            return ix, iy
        return None

    def zoom_at(self, x, y, factor, shape, width, height):
        point = self.image_point(x, y, shape, width, height)
        if point is None:
            return
        self.zoom = min(32.0, max(0.25, self.zoom * factor))
        scale, _, _ = self.transform(shape, width, height)
        self.center_x = (point[0] - (x - width / 2) / scale) / shape[1]
        self.center_y = (point[1] - (y - height / 2) / scale) / shape[0]


def surface_values_from_points(points, width):
    (x1, y1), (x2, y2) = points
    if abs(x2 - x1) < 1e-6:
        raise ValueError("Choose two points with different x coordinates.")
    slope = (y2 - y1) / (x2 - x1)
    center_y = y1 + slope * ((width - 1) / 2 - x1)
    if center_y < 0:
        raise ValueError("The surface line must have a non-negative center y.")
    return round(center_y), math.degrees(math.atan(slope))


class CollapsibleGroup(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title = title
        self.expanded = True
        self.header = ttk.Button(self, text=f"- {title}", command=self.toggle, style="Bold.TButton")
        self.header.pack(fill="x", pady=(8, 4))
        self.body = ttk.Frame(self)
        self.body.pack(fill="x")

    def toggle(self):
        self.expanded = not self.expanded
        self.header.configure(text=f"{'-' if self.expanded else '+'} {self.title}")
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


def responsive_label(parent, **options) -> ttk.Label:
    label = ttk.Label(parent, wraplength=1, justify="left", **options)
    parent.bind("<Configure>", lambda event: label.configure(wraplength=max(1, event.width)), add="+")
    return label


class ParameterControl(ttk.Frame):
    def __init__(self, parent, spec: Parameter, changed: Callable[[], None]):
        super().__init__(parent, padding=(0, 4))
        self.spec, self.changed = spec, changed
        self.updating = False
        self.value = tk.StringVar()
        self.auto = tk.BooleanVar()
        self.valid = True
        self.initial = None
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text=spec.label, wraplength=195).pack(side="left")
        self.reset_button = ttk.Button(header, text="Reset", width=6, command=self.reset_value)
        self.reset_button.pack(side="right", padx=(4, 0))
        if spec.automatic:
            ttk.Checkbutton(header, text="Auto", variable=self.auto,
                            command=self._auto_changed).pack(side="right")
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.slider_value = tk.DoubleVar(value=spec.low)
        self.slider = ttk.Scale(row, from_=spec.low, to=spec.high,
                                variable=self.slider_value, command=self._slide)
        self.slider.bind("<Button-1>", self._slider_press)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry = ttk.Entry(row, textvariable=self.value, width=10)
        self.entry.pack(side="right")
        self.entry.bind("<Return>", lambda _: self._typed())
        self.entry.bind("<FocusOut>", lambda _: self._typed())
        self.value.trace_add("write", lambda *_: self._typed())
        responsive_label(self, text=spec.description,
                         foreground="#526071").pack(fill="x", pady=(2, 0))
        self.error = responsive_label(self, foreground="#b42318")
        self.error.pack(fill="x")

    def set_value(self, value: float | None, resolved: float, remember: bool = True):
        if remember:
            self.initial = (value, resolved)
        self.updating = True
        self.auto.set(value is None)
        display = resolved if value is None else value
        # Preserve configured values even beyond the suggested slider range.
        self.slider.configure(from_=min(self.spec.low, display), to=max(self.spec.high, display))
        # Scale.set ignores updates while disabled (during loading or Auto).
        self.slider_value.set(display)
        self.value.set(str(display))
        self.updating = False
        self.valid = True
        self.error.configure(text="")
        self._state()

    def reset_value(self):
        if self.initial is not None:
            self.set_value(*self.initial)
            self.changed()

    def _state(self):
        state = "disabled" if self.auto.get() else "normal"
        self.slider.configure(state=state)
        self.entry.configure(state=state)

    def _auto_changed(self):
        self._state()
        self._typed()

    def _slider_press(self, event):
        if self.slider.instate(["disabled"]):
            return "break"
        if self.slider.identify(event.x, event.y).endswith("slider"):
            return None
        # Let the theme map track coordinates to values, including thumb margins.
        self.slider.set(self.slider.get(event.x, event.y))
        return "break"

    def _slide(self, value):
        if self.updating:
            return
        number = round(float(value) / self.spec.step) * self.spec.step
        self.value.set(f"{number:g}")

    def _typed(self):
        if self.updating:
            return
        try:
            number = self.get_value()
            if number is not None:
                self.updating = True
                self.slider.configure(from_=min(self.spec.low, number), to=max(self.spec.high, number))
                self.slider_value.set(number)
                self.updating = False
            self.error.configure(text="")
            self.valid = True
        except ValueError as error:
            self.error.configure(text=str(error))
            self.valid = False
        self.changed()

    def get_value(self) -> float | int | None:
        if self.spec.automatic and self.auto.get():
            return None
        try:
            number = float(self.value.get())
        except ValueError:
            raise ValueError("Enter a valid number.") from None
        if not math.isfinite(number):
            raise ValueError("Enter a finite value.")
        if self.spec.name in ("strong_threshold", "weak_threshold", "min_strong_threshold", "weak_threshold_ratio", "fps", "pixel_size_mm_per_px"):
            if number <= 0:
                raise ValueError("Enter a positive value.")
        elif self.spec.name != "surface_angle_deg" and number < self.spec.low:
            raise ValueError(f"Enter a value >= {self.spec.low:g}.")
        if self.spec.name in ("weak_threshold_ratio", "top_exclusion_ratio") and number > 1:
            raise ValueError("Ratio must be <= 1.")
        if self.spec.name in ("surface_y_px", "weak_reach_px", "min_droplet_area_px2",
                              "morphology_radius_px", "surface_band_px", "foreground_depth_px",
                              "min_contact_px", "surface_calibration_frame", "droplet_size_ref_frame", "mirror_min_depth_px"):
            if not number.is_integer():
                raise ValueError("Enter a whole number.")
            return int(number)
        return number


class DropletApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        # Block the default combobox wheel bindings; clicks and keys still work.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind_class("TCombobox", sequence, lambda _: "break")
        root.title("Droplet Impact CV - Semi-automatic Mode")
        root.geometry("1380x900")
        root.minsize(1000, 650)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.messages: queue.Queue = queue.Queue()
        self.closed = threading.Event()
        self.generation = 0
        self.session: PreviewSession | None = None
        self.pending = None
        self.preview_future = None
        self.processing = False
        self.preview_ready = False
        self.recalibrate_requested = False
        self.play_direction = 0
        self.play_timer = None
        self.preview_started = 0.0
        self.playback_fps = tk.StringVar(value="5")
        self.images = None
        self.photos = []
        self.viewports = [PreviewViewport(), PreviewViewport()]
        self.selecting_surface = False
        self.surface_points = []
        self.selection_frame_index = None
        self.displayed_frame_index = None
        self.controls = {}
        self.status = tk.StringVar(value="Choose one folder to start.")
        self.folder = tk.StringVar(value="No folder selected")
        self.frame_text = tk.StringVar(value="Frame: -")
        self.details = tk.StringVar()
        self.reflection = tk.StringVar(value="auto")
        self.rim_to_rim = tk.BooleanVar()
        self.shadow_suppression = tk.BooleanVar(value=True)
        self.output_mode = tk.StringVar(value="input")
        self.custom_output = tk.StringVar()
        self.output_summary = tk.StringVar(value="Choose an input folder first.")
        self._build()
        self._lock_parameters(True)
        self._lock_output_controls()
        self._lock_frame_controls()
        self.custom_output.trace_add("write", lambda *_: self.output_changed())
        self.playback_fps.trace_add("write", lambda *_: self.playback_speed_changed())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_timer = root.after(50, self._poll)

    def _build(self):
        self.bold_font = font.nametofont("TkDefaultFont").copy()
        self.bold_font.configure(weight="bold")
        style = ttk.Style(self.root)
        style.configure("Bold.TButton", font=self.bold_font)
        style.configure("Playback.TButton", font=("Segoe UI Symbol", 12))
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        self.choose = ttk.Button(top, text="Choose Folder...", command=self.choose_folder)
        self.choose.pack(side="left")
        self.process = ttk.Button(top, text="Process All Frames", command=self.process_all, state="disabled", style="Bold.TButton")
        self.process.pack(side="right")
        self.reset = ttk.Button(top, text="Reset Parameters", command=self.reset_parameters, state="disabled", style="Bold.TButton")
        self.reset.pack(side="right", padx=8)
        self.save = ttk.Button(top, text="Save cv_config.txt", command=self.save_parameters,
                               state="disabled", style="Bold.TButton")
        self.save.pack(side="right")
        responsive_label(top, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=12)
        output_panel = ttk.LabelFrame(self.root, text="Output Folder", padding=8)
        output_panel.pack(fill="x", padx=12, pady=(0, 8))
        modes = ttk.Frame(output_panel)
        modes.pack(fill="x")
        self.output_radios = []
        for text, value in (
            ("Project outputs (same as automatic mode)", "automatic"),
            ("Input folder / CV postprocessing", "input"),
            ("Custom folder", "custom"),
        ):
            radio = ttk.Radiobutton(modes, text=text, variable=self.output_mode,
                                    value=value, command=self.output_changed)
            radio.pack(side="left", padx=(0, 18))
            self.output_radios.append(radio)
        custom_row = ttk.Frame(output_panel)
        custom_row.pack(fill="x", pady=5)
        self.output_entry = ttk.Entry(custom_row, textvariable=self.custom_output)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_browse = ttk.Button(custom_row, text="Browse...", command=self.choose_output_folder)
        self.output_browse.pack(side="left", padx=(8, 0))
        ttk.Label(output_panel, textvariable=self.output_summary,
                  wraplength=1250).pack(anchor="w")
        content = ttk.Panedwindow(self.root, orient="horizontal")
        content.pack(fill="both", expand=True, padx=12)
        left = ttk.Frame(content, width=365)
        content.add(left, weight=0)
        ttk.Label(left, text="Analysis Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        canvas = tk.Canvas(left, width=350, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        self.parameter_panel = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.parameter_panel, anchor="nw")
        self.parameter_panel.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")))
        canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
        self.groups = {}
        specs = {spec.name: spec for spec in PARAMETERS}
        for title, names in PARAMETER_GROUPS:
            group = CollapsibleGroup(self.parameter_panel, title)
            group.pack(fill="x", padx=(0, 10))
            self.groups[title] = group
            if title == "Substrate and width measurement":
                self.select_surface_button = ttk.Button(
                    group.body, text="Select surface line manually", command=self.start_surface_selection,
                )
                self.select_surface_button.pack(fill="x", pady=(0, 8))
            for name in names:
                control = ParameterControl(group.body, specs[name], self.schedule_preview)
                control.pack(fill="x")
                self.controls[name] = control
        surface_group = self.groups["Substrate and width measurement"].body
        row = ttk.Frame(surface_group)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Reflection mode").pack(side="left")
        ttk.Button(row, text="Reset", width=6,
                   command=lambda: self.reset_choice("reflection_mode")).pack(side="right")
        self.reflection_combo = ttk.Combobox(surface_group, textvariable=self.reflection,
                             values=("auto", "mirror", "none"), state="readonly")
        self.reflection_combo.pack(fill="x")
        self.reflection_combo.bind("<<ComboboxSelected>>", lambda _: self.schedule_preview())
        responsive_label(surface_group, text="How to locate the substrate: mirror uses the drop and its reflection; none uses the drop's lower edge. Choose none if the reflection is faint. Auto selects a method.").pack(fill="x", pady=4)
        row = ttk.Frame(surface_group)
        row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row, text="Rim-to-rim width",
                        variable=self.rim_to_rim, command=self.schedule_preview).pack(side="left")
        ttk.Button(row, text="Reset", width=6,
                   command=lambda: self.reset_choice("rim_to_rim")).pack(side="right")
        responsive_label(surface_group, text="Use the widest rim-to-rim span. Turn off to measure where the drop meets the substrate.").pack(fill="x", pady=4)
        liquid_group = self.groups["Liquid detection"].body
        row = ttk.Frame(liquid_group)
        row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row, text="Suppress near-surface shadow",
                        variable=self.shadow_suppression, command=self.schedule_preview).pack(side="left")
        ttk.Button(row, text="Reset", width=6,
                   command=lambda: self.reset_choice("shadow_suppression")).pack(side="right")
        responsive_label(liquid_group, text="Use the drop contour and two preceding frames to limit shadows before and at initial contact. Sharp thin sheets and deformed drops retain faint-liquid growth. Check contact-onset warnings.").pack(fill="x", pady=4)
        right = ttk.Frame(content, padding=(12, 0, 0, 0))
        content.add(right, weight=1)
        self.preview_panel = ttk.Frame(right)
        self.preview_panel.pack(fill="both", expand=True)
        self.preview_labels = []
        for index, title in enumerate(("Before Processing", "After Processing")):
            pane = ttk.LabelFrame(self.preview_panel, text=title, padding=4)
            pane.grid(row=0, column=index, sticky="nsew", padx=4)
            self.preview_panel.columnconfigure(index, weight=1, uniform="preview")
            label = tk.Canvas(pane, width=1, height=1, highlightthickness=0, background="#20252b")
            label.pack(fill="both", expand=True)
            label.bind("<Configure>", lambda _: self.draw_images())
            label.bind("<MouseWheel>", lambda event, i=index: self.zoom_image(event, i))
            label.bind("<Button-4>", lambda event, i=index: self.zoom_image(event, i, 1))
            label.bind("<Button-5>", lambda event, i=index: self.zoom_image(event, i, -1))
            ttk.Button(label, text="Reset size", command=lambda i=index: self.reset_image_size(i)).place(relx=1, x=-6, y=6, anchor="ne")
            self.preview_labels.append(label)
        self.preview_labels[0].bind("<Button-1>", self.select_surface_point)
        self.surface_selection_panel = ttk.Frame(self.preview_labels[0], padding=6)
        self.surface_hint = tk.StringVar()
        responsive_label(self.surface_selection_panel, textvariable=self.surface_hint).pack(fill="x")
        row = ttk.Frame(self.surface_selection_panel)
        row.pack(fill="x", pady=(4, 0))
        self.surface_confirm = ttk.Button(row, text="Confirm", command=self.confirm_surface_selection)
        self.surface_confirm.pack(side="left")
        ttk.Button(row, text="Reselect", command=self.reselect_surface).pack(side="left", padx=6)
        self.preview_panel.rowconfigure(0, weight=1)
        self.preview_panel.bind("<Configure>", lambda _: self.draw_images())
        ttk.Label(right, textvariable=self.frame_text).pack(anchor="w", pady=(8, 0))
        frame_row = ttk.Frame(right)
        frame_row.pack(fill="x", pady=4)
        self.frame_slider = ttk.Scale(frame_row, from_=0, to=0, command=self.frame_changed, state="disabled")
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.frame_buttons = []
        for label, command in (
            ("⏮", lambda: self.navigate_frame("first")),
            ("|◀", lambda: self.navigate_frame(-1)),
            ("◀", lambda: self.play(-1)),
            ("⏸", self.pause),
            ("▶", lambda: self.play(1)),
            ("▶|", lambda: self.navigate_frame(1)),
            ("⏭", lambda: self.navigate_frame("last")),
        ):
            button = ttk.Button(frame_row, text=label, command=command, width=3, style="Playback.TButton")
            button.pack(side="left", padx=1)
            self.frame_buttons.append(button)
        ttk.Label(frame_row, text="Playback fps").pack(side="left", padx=(8, 4))
        self.speed_entry = ttk.Entry(frame_row, textvariable=self.playback_fps, width=6)
        self.speed_entry.pack(side="left")
        self.speed_reset = ttk.Button(frame_row, text="Reset", width=5, command=lambda: self.playback_fps.set("5"))
        self.speed_reset.pack(side="left", padx=(4, 0))
        ttk.Label(right, textvariable=self.details, wraplength=850).pack(anchor="w")
        self.range_label = ttk.Label(right, text="Working range: -")
        self.range_label.pack(anchor="w", pady=4)
        bottom = ttk.Frame(self.root, padding=12)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill="x")
        self.status_label = ttk.Label(bottom, textvariable=self.status, foreground=STATUS_COLORS["idle"])
        self.status_label.pack(anchor="w", pady=(5, 0))

    def set_status(self, text: str, kind: str | None = None):
        if kind is None:
            if text.startswith(("Ready.", "Complete.")):
                kind = "ready"
            elif text.startswith("Error:"):
                kind = "error"
            elif text.startswith(("Loading", "Finding", "Updating", "Processing")):
                kind = "busy"
            else:
                kind = "idle"
        self.status.set(text)
        self.status_label.configure(foreground=STATUS_COLORS[kind])

    def _lock_frame_controls(self):
        state = "normal" if self.session is not None and not self.processing else "disabled"
        for button in self.frame_buttons:
            button.configure(state=state)
        self.speed_entry.configure(state=state)
        self.speed_reset.configure(state=state)

    def _playback_rate(self) -> float:
        try:
            rate = float(self.playback_fps.get())
        except ValueError:
            raise ValueError("Playback fps must be a positive number.") from None
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Playback fps must be a finite positive number.")
        return rate

    def playback_speed_changed(self):
        try:
            self._playback_rate()
        except ValueError as error:
            self.pause()
            self.set_status(f"Error: {error}", "error")
            return
        if self.play_direction:
            if self.play_timer is not None:
                self.root.after_cancel(self.play_timer)
                self.play_timer = None
            self._schedule_playback_step()
        elif self.preview_ready:
            self.set_status("Ready. Adjust parameters or process all frames.")

    def pause(self):
        was_playing = self.play_direction != 0
        self.play_direction = 0
        if self.play_timer is not None:
            self.root.after_cancel(self.play_timer)
            self.play_timer = None
        if was_playing and self.preview_ready:
            self.set_status("Ready. Preview playback paused.")

    def navigate_frame(self, destination: str | int):
        if self.session is None or self.processing:
            return
        self.pause()
        last = len(self.session.prepared.files) - 1
        current = round(self.frame_slider.get())
        index = 0 if destination == "first" else last if destination == "last" else current + destination
        self.frame_slider.set(min(last, max(0, index)))

    def play(self, direction: int):
        if self.session is None or self.processing:
            return
        try:
            self._playback_rate()
            self.overrides()
        except ValueError as error:
            self.pause()
            self.set_status(f"Error: {error}", "error")
            return
        self.pause()
        self.play_direction = direction
        self.set_status("Playing forward..." if direction > 0 else "Playing in reverse...", "busy")
        if self.preview_ready:
            self._playback_step()

    def _playback_step(self):
        self.play_timer = None
        if not self.play_direction or self.session is None or self.processing:
            return
        index = round(self.frame_slider.get()) + self.play_direction
        if not 0 <= index < len(self.session.prepared.files):
            self.pause()
            self.set_status("Ready. Reached the end of the working range.")
            return
        self.frame_slider.set(index)

    def _schedule_playback_step(self):
        if not self.play_direction or not self.preview_ready or self.play_timer is not None:
            return
        try:
            period = 1.0 / self._playback_rate()
        except ValueError as error:
            self.pause()
            self.set_status(f"Error: {error}", "error")
            return
        delay = max(1, round(1000 * (period - (time.monotonic() - self.preview_started))))
        self.play_timer = self.root.after(delay, self._playback_step)

    def _submit(self, tag: str, generation: int, work: Callable):
        def run():
            try:
                result = work()
                self.messages.put((tag, generation, result))
            except Exception as error:
                critical = tag == "processed" or not isinstance(error, ValueError)
                self.messages.put(("error", generation, (str(error), critical)))
        return self.executor.submit(run)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose One Image Folder", mustexist=True)
        if not folder:
            return
        self.pause()
        self.end_surface_selection()
        self.viewports = [PreviewViewport(), PreviewViewport()]
        self.displayed_frame_index = None
        self.generation += 1
        generation = self.generation
        self.session = None
        self.recalibrate_requested = False
        self.preview_ready = False
        self._lock_frame_controls()
        self._lock_output_controls()
        self.output_summary.set("Loading input folder...")
        self._lock_parameters(True)
        self._cancel_preview()
        self.images = None
        self.draw_images()
        self.process.configure(state="disabled")
        self.reset.configure(state="disabled")
        self.save.configure(state="disabled")
        self.frame_slider.configure(state="disabled")
        self.folder.set(folder)
        self.details.set("")
        self.frame_text.set("Frame: -")
        self.range_label.configure(text="Working range: -")
        self.set_status("Loading background and calibration...")
        self.choose.configure(state="disabled")
        self._submit("loaded", generation, lambda: PreviewSession.open(
            Path(folder), lambda text: self.messages.put(("status", generation, text)),
            self.closed.is_set,
        ))

    def _lock_output_controls(self):
        enabled = self.session is not None and not self.processing
        for radio in self.output_radios:
            radio.configure(state="normal" if enabled else "disabled")
        custom = enabled and self.output_mode.get() == "custom"
        self.output_entry.configure(state="normal" if custom else "disabled")
        self.output_browse.configure(state="normal" if custom else "disabled")

    def batch_config(self) -> AnalysisConfig:
        return resolve_output_config(
            replace(self.session.base_config, **self.overrides()),
            self.output_mode.get(), self.custom_output.get(),
        )

    def _update_process_state(self):
        self.save.configure(state="normal" if self.session is not None and not self.processing
                            and not self.selecting_surface else "disabled")
        ready = self.preview_ready and self.session is not None and not self.processing and not self.selecting_surface
        if ready:
            try:
                self.batch_config()
            except ValueError:
                ready = False
        self.process.configure(state="normal" if ready else "disabled")

    def output_changed(self):
        self._lock_output_controls()
        if self.session is None:
            return
        try:
            config = resolve_output_config(
                self.session.base_config, self.output_mode.get(), self.custom_output.get(),
            )
            self.output_summary.set(
                f"CSV: {config.output_csv} | Run parameters: {config.output_dir / RUN_PARAMS_FILENAME} | "
                f"Debug overlays: {config.debug_dir}"
            )
            if self.preview_ready:
                self.batch_config()
                self._playback_rate()
                self.set_status("Ready. Adjust parameters or process all frames.")
        except ValueError as error:
            self.output_summary.set(str(error))
            self.set_status(f"Error: {error}", "error")
        self._update_process_state()

    def choose_output_folder(self):
        folder = filedialog.askdirectory(title="Choose Output Folder", mustexist=False)
        if folder:
            self.output_mode.set("custom")
            self.custom_output.set(folder)
            self.output_changed()

    def _initialize_output(self):
        config = self.session.base_config
        folder = (config.output_dir or config.output_csv.parent).resolve()
        args = parse_args([str(config.input_dir)])
        configured = any(getattr(args, name) is not None for name in ("output_dir", "output_csv", "debug_dir"))
        standard = (config.output_csv.resolve() == folder / OUTPUT_CSV_FILENAME
                    and config.debug_dir is not None
                    and config.debug_dir.resolve() == folder / "debug_overlays")
        mode = "input"
        if configured:
            mode = ("input" if standard and folder == config.input_dir.resolve() / "CV_postprocessing"
                    else "automatic" if standard and folder == default_output_dir(config.input_dir).resolve()
                    else "custom")
        self.output_mode.set(mode)
        self.custom_output.set(str(folder))
        self.output_changed()

    def _set_controls(self):
        session = self.session
        if session is None:
            return
        resolved = {
            "strong_threshold": session.prepared.threshold,
            "weak_threshold": session.prepared.weak_threshold,
            "surface_y_px": session.prepared.surface_line.center_y_int(),
            "surface_angle_deg": session.prepared.surface_line.angle_deg,
            "surface_calibration_frame": frame_number_from_filename(session.prepared.files[-1]),
            "droplet_size_ref_frame": (session.droplet_size_ref_frame
                                             if session.droplet_size_ref_frame is not None
                                             else frame_number_from_filename(session.prepared.files[0])),
        }
        for name, control in self.controls.items():
            value = getattr(session.base_config, name)
            control.set_value(value, resolved.get(name, value))
        self.reflection.set(session.base_config.reflection_mode)
        self.rim_to_rim.set(session.base_config.rim_to_rim)
        self.shadow_suppression.set(session.base_config.shadow_suppression)

    def overrides(self) -> dict:
        result = {name: control.get_value() for name, control in self.controls.items()}
        result.update(reflection_mode=self.reflection.get(), rim_to_rim=self.rim_to_rim.get())
        result["shadow_suppression"] = self.shadow_suppression.get()
        frames = {frame_number_from_filename(path) for path in self.session.prepared.files}
        for name, label in (("surface_calibration_frame", "Surface calibration frame"),
                            ("droplet_size_ref_frame", "Droplet size reference frame")):
            if result[name] is not None and result[name] not in frames:
                raise ValueError(f"{label} must be an existing image number in the working range.")
        return result

    def reset_choice(self, name: str):
        if self.session is None or self.processing:
            return
        variable = {"reflection_mode": self.reflection, "rim_to_rim": self.rim_to_rim,
                    "shadow_suppression": self.shadow_suppression}[name]
        variable.set(getattr(self.session.base_config, name))
        self.schedule_preview()

    def reset_parameters(self):
        self.end_surface_selection()
        self._set_controls()
        self.schedule_preview()

    def save_parameters(self):
        if self.session is None or self.processing or self.selecting_surface:
            return
        try:
            config = self.batch_config()
            path = config.input_dir / CONFIG_FILENAME
            backup = messagebox.askyesnocancel(
                "Save cv_config.txt",
                "Back up the existing cv_config.txt before saving?\n\n"
                "Yes: rename it to cv_config.txt.bak (replace any previous backup).\n"
                "No: overwrite without a backup.\nCancel: do not save.",
                parent=self.root,
            )
            if backup is None:
                return
            save_folder_config(config, backup=backup)
        except (OSError, ValueError) as error:
            self.set_status(f"Error: {error}", "error")
            messagebox.showerror("Save Configuration Error", str(error), parent=self.root)
            return
        self.set_status(f"Complete. Saved current parameters to {path}")

    def frame_changed(self, value):
        if self.session is None or self.processing:
            return
        index = min(len(self.session.prepared.files) - 1, max(0, round(float(value))))
        number = frame_number_from_filename(self.session.prepared.files[index])
        if self.selecting_surface and index != self.selection_frame_index:
            self.reselect_surface()
        self.frame_text.set(f"Frame: {number} ({index + 1} / {len(self.session.prepared.files)})")
        self.schedule_preview()

    def _cancel_preview(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        if self.preview_future is not None:
            self.preview_future.cancel()

    def schedule_preview(self):
        if self.session is None or self.processing:
            return
        if self.play_timer is not None:
            self.root.after_cancel(self.play_timer)
            self.play_timer = None
        self.generation += 1
        self.preview_ready = False
        self._cancel_preview()
        if not self.selecting_surface:
            try:
                self.overrides()
            except ValueError as error:
                self.pause()
                self.process.configure(state="disabled")
                self.set_status(f"Error: {error} Correct the parameter to continue.", "error")
                return
        self.process.configure(state="disabled")
        self.set_status("Updating current frame...")
        self.pending = self.root.after(0 if self.play_direction else 200, self._preview)

    def _preview(self):
        self.pending = None
        self.preview_started = time.monotonic()
        index = min(len(self.session.prepared.files) - 1, max(0, round(self.frame_slider.get())))
        self.requested_preview_index = index
        session = self.session
        if self.selecting_surface:
            self.preview_future = self._submit("raw_preview", self.generation,
                                               lambda: session.raw_preview(index))
            return
        overrides = self.overrides()
        recalibrate = self.recalibrate_requested
        self.preview_future = self._submit("preview", self.generation,
                                           lambda: session.preview(index, overrides, recalibrate))

    def zoom_image(self, event, index, direction=None):
        if self.images is not None:
            canvas = self.preview_labels[index]
            delta = direction if direction is not None else event.delta / 120
            self.viewports[index].zoom_at(
                event.x, event.y, 1.2 ** max(-4, min(4, delta)), self.images[index].shape,
                max(1, canvas.winfo_width()), max(1, canvas.winfo_height()),
            )
            self.draw_images()
        return "break"

    def reset_image_size(self, index):
        self.viewports[index] = PreviewViewport()
        self.draw_images()

    def start_surface_selection(self):
        if self.session is None or self.processing:
            return
        self.pause()
        self.selecting_surface = True
        self.surface_selection_panel.place(relx=0, rely=1, relwidth=1, anchor="sw")
        self.preview_labels[0].configure(cursor="crosshair")
        self.reselect_surface()
        self._update_process_state()
        self.schedule_preview()

    def reselect_surface(self):
        self.surface_points = []
        self.selection_frame_index = None
        self.surface_hint.set("Select two points in Before Processing. You can change frame or zoom.")
        self.surface_confirm.configure(state="disabled")
        self.draw_images()

    def end_surface_selection(self):
        self.selecting_surface = False
        self.surface_points = []
        self.selection_frame_index = None
        self.surface_selection_panel.place_forget()
        self.preview_labels[0].configure(cursor="")
        self.draw_images()

    def select_surface_point(self, event):
        if (not self.selecting_surface or self.images is None or len(self.surface_points) == 2
                or self.displayed_frame_index != round(self.frame_slider.get())
                or self.pending is not None
                or (self.preview_future is not None and not self.preview_future.done())):
            return
        canvas = self.preview_labels[0]
        point = self.viewports[0].image_point(
            event.x, event.y, self.images[0].shape, max(1, canvas.winfo_width()), max(1, canvas.winfo_height()),
        )
        if point is None:
            return
        self.pause()
        self.selection_frame_index = self.displayed_frame_index
        self.surface_points.append(point)
        self.surface_hint.set("Select the second point." if len(self.surface_points) == 1 else "Confirm this line, or reselect the two points.")
        if len(self.surface_points) == 2:
            try:
                surface_values_from_points(self.surface_points, self.images[0].shape[1])
                self.surface_confirm.configure(state="normal")
            except ValueError as error:
                self.surface_hint.set(str(error) + " Reselect the points.")
        self.draw_images()

    def confirm_surface_selection(self):
        if (not self.selecting_surface or len(self.surface_points) != 2
                or self.selection_frame_index != self.displayed_frame_index
                or self.selection_frame_index != round(self.frame_slider.get())):
            return
        try:
            center_y, angle = surface_values_from_points(self.surface_points, self.images[0].shape[1])
        except ValueError as error:
            self.surface_hint.set(str(error) + " Reselect the points.")
            return
        frame = frame_number_from_filename(self.session.prepared.files[self.selection_frame_index])
        keep_auto_tilt = self.controls["surface_angle_deg"].auto.get() and messagebox.askyesno(
            "Surface Tilt", "Keep Surface tilt on Auto?\n\n"
            "Yes: detect the tilt automatically using the selected calibration frame.\n"
            "No: turn off Auto and use the tilt of the manually selected line.",
            parent=self.root,
        )
        for name, value in (("surface_calibration_frame", frame), ("surface_y_px", center_y), ("surface_angle_deg", angle)):
            control = self.controls[name]
            control.set_value(None if name == "surface_angle_deg" and keep_auto_tilt else value,
                              value, remember=False)
        self.end_surface_selection()
        self.recalibrate_requested = True
        self.schedule_preview()

    def draw_images(self):
        self.photos = []
        if self.images is None:
            for label in self.preview_labels:
                label.delete("all")
                label.create_text(max(1, label.winfo_width()) / 2, max(1, label.winfo_height()) / 2,
                                  text="No image", fill="white")
            return
        for index, (label, array) in enumerate(zip(self.preview_labels, self.images)):
            width, height = max(1, label.winfo_width()), max(1, label.winfo_height())
            scale, left, top = self.viewports[index].transform(array.shape, width, height)
            resized = cv2.warpAffine(array, np.array([[scale, 0, left], [0, scale, top]]),
                                     (width, height), flags=cv2.INTER_LINEAR, borderValue=(32, 37, 43))
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            header = f"P6\n{resized.shape[1]} {resized.shape[0]}\n255\n".encode("ascii")
            photo = tk.PhotoImage(data=header + resized.tobytes(), format="PPM")
            self.photos.append(photo)
            label.delete("all")
            label.create_image(0, 0, image=photo, anchor="nw")
            if index == 0 and self.selecting_surface:
                points = [(left + x * scale, top + y * scale) for x, y in self.surface_points]
                if len(points) == 2:
                    label.create_line(*points[0], *points[1], fill="#00e5ff", width=2)
                for x, y in points:
                    label.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00e5ff", width=2)

    def _lock_parameters(self, locked: bool):
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Scale, ttk.Checkbutton, ttk.Combobox, ttk.Button)):
                    child.configure(state="disabled" if locked else "normal")
                walk(child)
        walk(self.parameter_panel)
        for group in self.groups.values():
            group.header.configure(state="normal")
        if not locked:
            for control in self.controls.values():
                control._state()
            self.reflection_combo.configure(state="readonly")

    def process_all(self):
        if self.session is None or self.processing or self.selecting_surface:
            return
        try:
            config = self.batch_config()
        except ValueError as error:
            self.set_status(f"Error: {error}", "error")
            return
        self.pause()
        self._cancel_preview()
        self.generation += 1
        generation = self.generation
        self.processing = True
        self._lock_frame_controls()
        self._lock_output_controls()
        self._lock_parameters(True)
        for button in (self.choose, self.save, self.reset, self.process):
            button.configure(state="disabled")
        self.frame_slider.configure(state="disabled")
        self.progress.configure(value=0)
        self.set_status("Processing all frames with current settings...")

        def run():
            def progress(done, total):
                if self.closed.is_set():
                    raise RuntimeError("Operation cancelled.")
                self.messages.put(("progress", generation, (done, total)))
            measurements = analyze_sequence(config, progress=progress)
            if self.closed.is_set():
                raise RuntimeError("Operation cancelled.")
            write_csv(config.output_csv, measurements)
            write_run_params(config.output_dir / RUN_PARAMS_FILENAME, config)
            return len(measurements), config.output_csv

        self._submit("processed", generation, run)

    def _finish_processing(self):
        self.processing = False
        self._lock_frame_controls()
        self._lock_output_controls()
        self._lock_parameters(self.session is None)
        self.choose.configure(state="normal")
        self.reset.configure(state="normal" if self.session else "disabled")
        self.save.configure(state="normal" if self.session else "disabled")
        self.frame_slider.configure(state="normal" if self.session else "disabled")

    def _poll(self):
        self.poll_timer = None
        if self.closed.is_set():
            return
        try:
            while True:
                tag, generation, result = self.messages.get_nowait()
                if generation != self.generation:
                    continue
                if tag == "loaded":
                    self.session = result
                    self._initialize_output()
                    self._set_controls()
                    self._lock_parameters(False)
                    self._lock_frame_controls()
                    files = result.prepared.files
                    self.frame_slider.configure(from_=0, to=max(0, len(files) - 1), state="normal")
                    self.frame_slider.set(result.initial_index)
                    self.range_label.configure(text=(
                        f"Working range: {frame_number_from_filename(files[0])} - "
                        f"{frame_number_from_filename(files[-1])} | "
                        f"Initial droplet size reference frame: {result.droplet_size_ref_frame if result.droplet_size_ref_frame is not None else 'not detected (first frame shown)'}"
                    ))
                    self.choose.configure(state="normal")
                    self.reset.configure(state="normal")
                    self.save.configure(state="normal")
                    self.frame_changed(result.initial_index)
                elif tag == "raw_preview":
                    before, after, details = result
                    self.images = before, after
                    self.displayed_frame_index = self.requested_preview_index
                    self.details.set(details)
                    self.draw_images()
                    self.set_status("Select two points on the surface in Before Processing.")
                elif tag == "preview":
                    self.recalibrate_requested = False
                    before, after, details = result
                    prepared = self.session.prepared
                    resolved = {
                        "strong_threshold": prepared.threshold,
                        "weak_threshold": prepared.weak_threshold,
                        "surface_y_px": prepared.surface_line.center_y_int(),
                        "surface_angle_deg": prepared.surface_line.angle_deg,
                        "surface_calibration_frame": frame_number_from_filename(prepared.files[-1]),
                        "droplet_size_ref_frame": (self.session.droplet_size_ref_frame
                                                   if self.session.droplet_size_ref_frame is not None
                                                   else frame_number_from_filename(prepared.files[0])),
                    }
                    for name, value in resolved.items():
                        if self.controls[name].auto.get():
                            self.controls[name].set_value(None, value, remember=False)
                    self.images = before, after
                    self.displayed_frame_index = self.requested_preview_index
                    self.details.set(details)
                    self.draw_images()
                    self.preview_ready = True
                    self._update_process_state()
                    try:
                        self.batch_config()
                        self._playback_rate()
                        if self.play_direction:
                            self.set_status("Playing forward..." if self.play_direction > 0 else "Playing in reverse...", "busy")
                        else:
                            self.set_status("Ready. Adjust parameters or process all frames.")
                    except ValueError as error:
                        self.set_status(f"Error: {error}", "error")
                    self._schedule_playback_step()
                elif tag == "status":
                    self.set_status(result)
                elif tag == "progress":
                    done, total = result
                    self.progress.configure(maximum=total, value=done)
                    self.set_status(f"Processing frames: {done} / {total}")
                elif tag == "processed":
                    self._finish_processing()
                    self._update_process_state()
                    count, path = result
                    self.set_status(f"Complete. Wrote {count} rows to {path}")
                elif tag == "error":
                    self.preview_ready = False
                    self._finish_processing()
                    self.process.configure(state="disabled")
                    self.pause()
                    text, critical = result
                    self.set_status(f"Error: {text}", "error")
                    if critical:
                        messagebox.showerror("Processing Error", text, parent=self.root)
        except queue.Empty:
            pass
        self.poll_timer = self.root.after(50, self._poll)

    def close(self):
        if self.poll_timer is not None:
            self.root.after_cancel(self.poll_timer)
            self.poll_timer = None
        self.pause()
        self.closed.set()
        self._cancel_preview()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    # No folder argument or analysis overrides: the folder is selected in GUI.
    argparse.ArgumentParser(description="Open the semi-automatic droplet analysis GUI.").parse_args()
    root = tk.Tk()
    DropletApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
