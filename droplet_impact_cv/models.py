from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_FPS = 8000.0
DEFAULT_PIXEL_SIZE_MM = 0.00711883341
DEFAULT_SURFACE_ANGLE_DEG = -0.6
DEFAULT_MIN_FOREGROUND_DELTA = 1500.0
DEFAULT_THRESHOLD_SAMPLE_FRAMES = 8
DEFAULT_SURFACE_BAND_PX = 3
DEFAULT_WEAK_PROPAGATION_PX = 40
DEFAULT_WEAK_THRESHOLD_FRACTION = 0.25
REFLECTION_MODES = ("auto", "mirror", "none")


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_csv: Path
    fps: float = DEFAULT_FPS
    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM
    surface_y: int | None = None
    surface_frame: int | None = None
    surface_angle_deg: float | None = None
    reflection_mode: str = "auto"
    threshold: float | None = None
    weak_threshold: float | None = None
    weak_threshold_fraction: float = DEFAULT_WEAK_THRESHOLD_FRACTION
    weak_propagation_px: int = DEFAULT_WEAK_PROPAGATION_PX
    min_foreground_delta: float = DEFAULT_MIN_FOREGROUND_DELTA
    min_area_px: int = 250
    morphology_radius_px: int = 3
    surface_search_start_px: int = 50
    surface_drop_delta: float = 100.0
    surface_band_px: int = DEFAULT_SURFACE_BAND_PX
    hydrophobic: bool = False
    measure_below_surface_px: int = 220
    touch_above_surface_px: int = 20
    touch_below_surface_px: int = 170
    min_touch_pixels: int = 30
    min_impact_width_px: int = 30
    include_pre_impact: bool = False
    time_zero: str = "impact"
    debug_dir: Path | None = Path("outputs/debug_overlays")
    debug_every: int = 1
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass(frozen=True)
class FrameMeasurement:
    frame_number: int
    filename: str
    time_ms: float
    diameter_px: float
    diameter_mm: float
    component_area_px: int
    surface_y: int
    impact_frame: int | None
    fps: float
    pixel_size_mm: float
    surface_frame: int | None


@dataclass(frozen=True)
class ComponentMeasurement:
    diameter_px: float
    area_px: int
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray
    contact_bbox: tuple[int, int, int, int] | None = None
    rim_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class SurfaceLine:
    center_y: float
    angle_deg: float = DEFAULT_SURFACE_ANGLE_DEG

    @property
    def slope(self) -> float:
        # Image y increases downward, so positive dy/dx is visually clockwise.
        return math.tan(math.radians(self.angle_deg))

    def y_at(self, x: np.ndarray | float, width: int) -> np.ndarray | float:
        center_x = (width - 1) / 2.0
        return self.center_y + self.slope * (x - center_x)

    def center_y_int(self) -> int:
        return int(round(self.center_y))
