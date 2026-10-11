from __future__ import annotations

import argparse
import math
import queue
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np

from .analysis import (
    PreparedSequence, analyze_sequence, is_reference_measurement, prepare_sequence,
)
from .cli import config_from_args, default_output_dir, parse_args
from .imaging import frame_number_from_filename
from .models import AnalysisConfig
from .output import write_csv
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
    return replace(
        config, output_csv=folder / config.output_csv.name,
        debug_dir=folder / "debug_overlays",
    )


@dataclass
class PreviewSession:
    base_config: AnalysisConfig
    prepared: PreparedSequence
    initial_index: int
    ref_frame: int | None
    applied_config: AnalysisConfig | None = None

    @classmethod
    def open(
        cls, folder: Path, report: Callable[[str], None] = lambda _: None,
        stopped: Callable[[], bool] = lambda: False,
    ) -> PreviewSession:
        config = load_folder_config(folder)
        prepared = prepare_sequence(config)
        # Reference discovery happens once at open, stopping at the first match.
        # It never writes CSVs or removes/writes diagnostic files.
        for index, path in enumerate(prepared.files):
            if stopped():
                raise RuntimeError("Operation cancelled.")
            report(f"Finding ref frame: {index + 1} / {len(prepared.files)}")
            _, measurement = prepared.measure(path)
            if is_reference_measurement(measurement):
                return cls(config, prepared, index, frame_number_from_filename(path))
        return cls(config, prepared, 0, None)

    def preview(self, index: int, overrides: dict) -> tuple[np.ndarray, np.ndarray, str]:
        config = replace(self.base_config, **overrides)
        if config != (self.applied_config or self.base_config):
            # Reuse the fixed background and file range; calibration and sampled
            # automatic thresholds follow the same rules as full analysis.
            self.prepared = prepare_sequence(
                config, self.prepared.files, self.prepared.background,
            )
            self.applied_config = config
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
    Parameter("strong_threshold", "Strong threshold", 1, 4095, 1,
              "Reliable foreground change. Lower values admit fainter liquid and more noise.", True),
    Parameter("min_strong_threshold", "Min strong threshold", 1, 4095, 1,
              "Lower bound for Auto strong threshold. Lower values increase sensitivity."),
    Parameter("weak_threshold", "Weak threshold", 1, 4095, 1,
              "Connected faint liquid. Lower values recover more transparent regions and noise.", True),
    Parameter("weak_threshold_ratio", "Weak / strong ratio", 0.001, 1, 0.001,
              "Used with Auto weak threshold. Lower ratios admit fainter connected liquid."),
    Parameter("weak_reach_px", "Weak reach (px)", 0, 200, 1,
              "Maximum growth from strong foreground. Increase to recover longer faint contours."),
    Parameter("min_droplet_area_px2", "Min droplet area (px2)", 1, 10000, 1,
              "Reject smaller components. Increase to remove small artifacts or droplets."),
    Parameter("morphology_radius_px", "Morphology radius (px)", 0, 15, 1,
              "Contour cleanup size. Increase to close gaps and remove small artifacts."),
    Parameter("surface_band_px", "Surface band (px)", 0, 30, 1,
              "Contact measurement half-height. Increase to tolerate blurred contact boundaries."),
    Parameter("foreground_depth_px", "Foreground depth (px)", 0, 500, 1,
              "Detection depth below the surface. Increase to include more lower foreground."),
    Parameter("min_contact_px", "Min contact pixels (px)", 0, 1000, 1,
              "Minimum gas-side pixels in the selected component. Increase to reject tiny regions."),
    Parameter("surface_y_px", "Surface center y (px)", 0, 2048, 1,
              "Override the fixed surface position. Larger y moves the surface downward.", True),
    Parameter("surface_angle_deg", "Surface angle (deg)", -45, 45, 0.1,
              "Override the surface tilt. Positive angles rotate clockwise.", True),
)


class ParameterControl(ttk.Frame):
    def __init__(self, parent, spec: Parameter, changed: Callable[[], None]):
        super().__init__(parent, padding=(0, 4))
        self.spec, self.changed = spec, changed
        self.updating = False
        self.value = tk.StringVar()
        self.auto = tk.BooleanVar()
        self.valid = True
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text=spec.label).pack(side="left")
        if spec.automatic:
            ttk.Checkbutton(header, text="Auto", variable=self.auto,
                            command=self._auto_changed).pack(side="right")
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.slider = ttk.Scale(row, from_=spec.low, to=spec.high, command=self._slide)
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry = ttk.Entry(row, textvariable=self.value, width=10)
        self.entry.pack(side="right")
        self.entry.bind("<Return>", lambda _: self._typed())
        self.entry.bind("<FocusOut>", lambda _: self._typed())
        self.value.trace_add("write", lambda *_: self._typed())
        ttk.Label(self, text=spec.description, wraplength=325,
                  foreground="#526071").pack(anchor="w", pady=(2, 0))
        self.error = ttk.Label(self, foreground="#b42318", wraplength=325)
        self.error.pack(anchor="w")

    def set_value(self, value: float | None, resolved: float):
        self.updating = True
        self.auto.set(value is None)
        display = resolved if value is None else value
        # Preserve configured values even beyond the suggested slider range.
        self.slider.configure(from_=min(self.spec.low, display), to=max(self.spec.high, display))
        self.slider.set(display)
        self.value.set(str(display))
        self.updating = False
        self.valid = True
        self.error.configure(text="")
        self._state()

    def _state(self):
        state = "disabled" if self.auto.get() else "normal"
        self.slider.configure(state=state)
        self.entry.configure(state=state)

    def _auto_changed(self):
        self._state()
        self._typed()

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
                self.slider.set(number)
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
        if self.spec.name in ("strong_threshold", "weak_threshold", "min_strong_threshold", "weak_threshold_ratio"):
            if number <= 0:
                raise ValueError("Enter a positive value.")
        elif self.spec.name != "surface_angle_deg" and number < self.spec.low:
            raise ValueError(f"Enter a value >= {self.spec.low:g}.")
        if self.spec.name == "weak_threshold_ratio" and number > 1:
            raise ValueError("Ratio must be <= 1.")
        if self.spec.name in ("surface_y_px", "weak_reach_px", "min_droplet_area_px2",
                              "morphology_radius_px", "surface_band_px", "foreground_depth_px",
                              "min_contact_px"):
            if not number.is_integer():
                raise ValueError("Enter a whole number.")
            return int(number)
        return number


class DropletApp:
    def __init__(self, root: tk.Tk):
        self.root = root
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
        self.images = None
        self.photos = []
        self.controls = {}
        self.status = tk.StringVar(value="Choose one folder to start.")
        self.folder = tk.StringVar(value="No folder selected")
        self.frame_text = tk.StringVar(value="Frame: -")
        self.details = tk.StringVar()
        self.reflection = tk.StringVar(value="auto")
        self.hydrophobic = tk.BooleanVar()
        self.output_mode = tk.StringVar(value="automatic")
        self.custom_output = tk.StringVar()
        self.output_summary = tk.StringVar(value="Choose an input folder first.")
        self._build()
        self._lock_parameters(True)
        self._lock_output_controls()
        self.custom_output.trace_add("write", lambda *_: self.output_changed())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self._poll)

    def _build(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        self.choose = ttk.Button(top, text="Choose Folder...", command=self.choose_folder)
        self.choose.pack(side="left")
        ttk.Label(top, textvariable=self.folder, wraplength=650).pack(side="left", padx=12)
        self.process = ttk.Button(top, text="Process All Frames", command=self.process_all, state="disabled")
        self.process.pack(side="right")
        self.reset = ttk.Button(top, text="Reset Parameters", command=self.reset_parameters, state="disabled")
        self.reset.pack(side="right", padx=8)
        output_panel = ttk.LabelFrame(self.root, text="Output Folder", padding=8)
        output_panel.pack(fill="x", padx=12, pady=(0, 8))
        modes = ttk.Frame(output_panel)
        modes.pack(fill="x")
        self.output_radios = []
        for text, value in (
            ("Project outputs (same as automatic mode)", "automatic"),
            ("Input folder / CV_postprocessing", "input"),
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
        ttk.Label(left, text="Contour Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w")
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
        for spec in PARAMETERS:
            control = ParameterControl(self.parameter_panel, spec, self.schedule_preview)
            control.pack(fill="x", padx=(0, 10))
            self.controls[spec.name] = control
        ttk.Label(self.parameter_panel, text="Reflection mode").pack(anchor="w", pady=(10, 0))
        combo = ttk.Combobox(self.parameter_panel, textvariable=self.reflection,
                             values=("auto", "mirror", "none"), state="readonly")
        combo.pack(fill="x", padx=(0, 10))
        combo.bind("<<ComboboxSelected>>", lambda _: self.schedule_preview())
        ttk.Label(self.parameter_panel, text="Surface calibration method. Use none for weak reflections.",
                  wraplength=325).pack(anchor="w", pady=4)
        ttk.Checkbutton(self.parameter_panel, text="Hydrophobic (rim-to-rim)",
                        variable=self.hydrophobic, command=self.schedule_preview).pack(anchor="w", pady=(8, 0))
        ttk.Label(self.parameter_panel, text="Measures full rim width instead of surface contact width.",
                  wraplength=325).pack(anchor="w", pady=4)
        right = ttk.Frame(content, padding=(12, 0, 0, 0))
        content.add(right, weight=1)
        self.preview_panel = ttk.Frame(right)
        self.preview_panel.pack(fill="both", expand=True)
        self.preview_labels = []
        for index, title in enumerate(("Before Processing", "After Processing")):
            pane = ttk.LabelFrame(self.preview_panel, text=title, padding=4)
            pane.grid(row=0, column=index, sticky="nsew", padx=4)
            self.preview_panel.columnconfigure(index, weight=1, uniform="preview")
            label = ttk.Label(pane, text="No image", anchor="center")
            label.pack(fill="both", expand=True)
            self.preview_labels.append(label)
        self.preview_panel.rowconfigure(0, weight=1)
        self.preview_panel.bind("<Configure>", lambda _: self.draw_images())
        ttk.Label(right, textvariable=self.frame_text).pack(anchor="w", pady=(8, 0))
        self.frame_slider = ttk.Scale(right, from_=0, to=0, command=self.frame_changed, state="disabled")
        self.frame_slider.pack(fill="x", pady=4)
        ttk.Label(right, textvariable=self.details, wraplength=850).pack(anchor="w")
        self.range_label = ttk.Label(right, text="Working range: -")
        self.range_label.pack(anchor="w", pady=4)
        bottom = ttk.Frame(self.root, padding=12)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(anchor="w", pady=(5, 0))

    def _submit(self, tag: str, generation: int, work: Callable):
        def run():
            try:
                result = work()
                self.messages.put((tag, generation, result))
            except Exception as error:
                self.messages.put(("error", generation, str(error)))
        return self.executor.submit(run)

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose One Image Folder", mustexist=True)
        if not folder:
            return
        self.generation += 1
        generation = self.generation
        self.session = None
        self.preview_ready = False
        self._lock_output_controls()
        self.output_summary.set("Loading input folder...")
        self._lock_parameters(True)
        self._cancel_preview()
        self.images = None
        self.draw_images()
        self.process.configure(state="disabled")
        self.reset.configure(state="disabled")
        self.frame_slider.configure(state="disabled")
        self.folder.set(folder)
        self.details.set("")
        self.frame_text.set("Frame: -")
        self.range_label.configure(text="Working range: -")
        self.status.set("Loading background and calibration...")
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
        ready = self.preview_ready and self.session is not None and not self.processing
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
                f"CSV: {config.output_csv} | Debug overlays: {config.debug_dir}"
            )
        except ValueError as error:
            self.output_summary.set(str(error))
        self._update_process_state()

    def choose_output_folder(self):
        folder = filedialog.askdirectory(title="Choose Output Folder", mustexist=False)
        if folder:
            self.output_mode.set("custom")
            self.custom_output.set(folder)
            self.output_changed()

    def _initialize_output(self):
        config = self.session.base_config
        folder = config.output_csv.parent.resolve()
        self.output_mode.set(
            "automatic" if folder == default_output_dir(config.input_dir) else "custom"
        )
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
        }
        for name, control in self.controls.items():
            value = getattr(session.base_config, name)
            control.set_value(value, resolved.get(name, value))
        self.reflection.set(session.base_config.reflection_mode)
        self.hydrophobic.set(session.base_config.hydrophobic)

    def overrides(self) -> dict:
        result = {name: control.get_value() for name, control in self.controls.items()}
        result.update(reflection_mode=self.reflection.get(), hydrophobic=self.hydrophobic.get())
        return result

    def reset_parameters(self):
        self._set_controls()
        self.schedule_preview()

    def frame_changed(self, value):
        if self.session is None or self.processing:
            return
        index = min(len(self.session.prepared.files) - 1, max(0, round(float(value))))
        number = frame_number_from_filename(self.session.prepared.files[index])
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
        self.generation += 1
        self.preview_ready = False
        self._cancel_preview()
        try:
            self.overrides()
        except ValueError:
            self.process.configure(state="disabled")
            self.status.set("Correct the highlighted parameter before previewing or processing.")
            return
        self.process.configure(state="disabled")
        self.status.set("Updating current frame...")
        self.pending = self.root.after(200, self._preview)

    def _preview(self):
        self.pending = None
        overrides = self.overrides()
        index = min(len(self.session.prepared.files) - 1, max(0, round(self.frame_slider.get())))
        session = self.session
        self.preview_future = self._submit("preview", self.generation,
                                           lambda: session.preview(index, overrides))

    def draw_images(self):
        self.photos = []
        if self.images is None:
            for label in self.preview_labels:
                label.configure(image="", text="No image")
            return
        width = max(1, self.preview_panel.winfo_width() // 2 - 24)
        height = max(1, self.preview_panel.winfo_height() - 40)
        for label, array in zip(self.preview_labels, self.images):
            scale = min(width / array.shape[1], height / array.shape[0])
            resized = cv2.resize(array, (max(1, int(array.shape[1] * scale)),
                                        max(1, int(array.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            header = f"P6\n{resized.shape[1]} {resized.shape[0]}\n255\n".encode("ascii")
            photo = tk.PhotoImage(data=header + resized.tobytes(), format="PPM")
            self.photos.append(photo)
            label.configure(image=photo, text="")

    def _lock_parameters(self, locked: bool):
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Scale, ttk.Checkbutton, ttk.Combobox)):
                    child.configure(state="disabled" if locked else "normal")
                walk(child)
        walk(self.parameter_panel)
        if not locked:
            for control in self.controls.values():
                control._state()
            for child in self.parameter_panel.winfo_children():
                if isinstance(child, ttk.Combobox):
                    child.configure(state="readonly")

    def process_all(self):
        if self.session is None or self.processing:
            return
        try:
            config = self.batch_config()
        except ValueError as error:
            self.status.set(str(error))
            return
        self._cancel_preview()
        self.generation += 1
        generation = self.generation
        self.processing = True
        self._lock_output_controls()
        self._lock_parameters(True)
        for button in (self.choose, self.reset, self.process):
            button.configure(state="disabled")
        self.frame_slider.configure(state="disabled")
        self.progress.configure(value=0)
        self.status.set("Processing all frames with current settings...")

        def run():
            def progress(done, total):
                if self.closed.is_set():
                    raise RuntimeError("Operation cancelled.")
                self.messages.put(("progress", generation, (done, total)))
            measurements = analyze_sequence(config, progress=progress)
            if self.closed.is_set():
                raise RuntimeError("Operation cancelled.")
            write_csv(config.output_csv, measurements)
            return len(measurements), config.output_csv

        self._submit("processed", generation, run)

    def _finish_processing(self):
        self.processing = False
        self._lock_output_controls()
        self._lock_parameters(self.session is None)
        self.choose.configure(state="normal")
        self.reset.configure(state="normal" if self.session else "disabled")
        self.frame_slider.configure(state="normal" if self.session else "disabled")

    def _poll(self):
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
                    files = result.prepared.files
                    self.frame_slider.configure(from_=0, to=max(0, len(files) - 1), state="normal")
                    self.frame_slider.set(result.initial_index)
                    self.range_label.configure(text=(
                        f"Working range: {frame_number_from_filename(files[0])} - "
                        f"{frame_number_from_filename(files[-1])} | "
                        f"Initial ref frame: {result.ref_frame if result.ref_frame is not None else 'not detected (first frame shown)'}"
                    ))
                    self.choose.configure(state="normal")
                    self.reset.configure(state="normal")
                    self.frame_changed(result.initial_index)
                elif tag == "preview":
                    before, after, details = result
                    prepared = self.session.prepared
                    resolved = {
                        "strong_threshold": prepared.threshold,
                        "weak_threshold": prepared.weak_threshold,
                        "surface_y_px": prepared.surface_line.center_y_int(),
                        "surface_angle_deg": prepared.surface_line.angle_deg,
                    }
                    for name, value in resolved.items():
                        if self.controls[name].auto.get():
                            self.controls[name].set_value(None, value)
                    self.images = before, after
                    self.details.set(details)
                    self.draw_images()
                    self.preview_ready = True
                    self._update_process_state()
                    self.status.set("Ready. Adjust parameters or process all frames.")
                elif tag == "status":
                    self.status.set(result)
                elif tag == "progress":
                    done, total = result
                    self.progress.configure(maximum=total, value=done)
                    self.status.set(f"Processing frames: {done} / {total}")
                elif tag == "processed":
                    self._finish_processing()
                    self._update_process_state()
                    count, path = result
                    self.status.set(f"Complete. Wrote {count} rows to {path}")
                elif tag == "error":
                    self.preview_ready = False
                    self._finish_processing()
                    self.process.configure(state="disabled")
                    self.status.set(f"Error: {result}")
                    messagebox.showerror("Processing Error", result, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def close(self):
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
