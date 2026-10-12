from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_FPS = 8000.0
DEFAULT_PIXEL_SIZE_MM_PER_PX = 0.00711883341
DEFAULT_SURFACE_ANGLE_DEG = -0.6
DEFAULT_MIN_STRONG_THRESHOLD = 1500.0
DEFAULT_THRESHOLD_SAMPLE_FRAMES = 8
DEFAULT_SURFACE_BAND_PX = 3
DEFAULT_WEAK_REACH_PX = 40
DEFAULT_WEAK_THRESHOLD_RATIO = 0.3
DEFAULT_MIRROR_MIN_DEPTH_PX = 50
DEFAULT_TOP_EXCLUSION_RATIO = 0.25
REFLECTION_MODES = ("auto", "mirror", "none")


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_csv: Path
    fps: float = DEFAULT_FPS
    pixel_size_mm_per_px: float = DEFAULT_PIXEL_SIZE_MM_PER_PX
    surface_y_px: int | None = None
    surface_calibration_frame: int | None = None
    droplet_size_ref_frame: int | None = None
    surface_angle_deg: float | None = None
    reflection_mode: str = "auto"
    mirror_min_depth_px: int = DEFAULT_MIRROR_MIN_DEPTH_PX
    strong_threshold: float | None = None
    weak_threshold: float | None = None
    weak_threshold_ratio: float = DEFAULT_WEAK_THRESHOLD_RATIO
    weak_reach_px: int = DEFAULT_WEAK_REACH_PX
    min_strong_threshold: float = DEFAULT_MIN_STRONG_THRESHOLD
    min_droplet_area_px2: int = 250
    morphology_radius_px: int = 3
    surface_search_start_px: int = 50
    surface_drop_delta: float = 100.0
    surface_band_px: int = DEFAULT_SURFACE_BAND_PX
    hydrophobic: bool = False
    foreground_depth_px: int = 220
    top_exclusion_ratio: float = DEFAULT_TOP_EXCLUSION_RATIO
    min_contact_px: int = 30
    min_impact_width_px: int = 30
    include_pre_impact: bool = False
    time_zero: str = "impact"
    debug_dir: Path | None = Path("outputs/debug_overlays")
    debug_interval: int = 1
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass(frozen=True)
class FrameMeasurement:
    frame_number: int
    filename: str
    time_ms: float
    d_spread_px: float
    d_spread_mm: float
    droplet_area_px2: int
    surface_y_px: int
    impact_frame: int | None
    fps: float
    pixel_size_mm_per_px: float
    surface_calibration_frame: int | None
    d_contact_px: float | None = None
    d_contact_mm: float | None = None
    d_rim_px: float | None = None
    d_rim_mm: float | None = None
    droplet_size_ref_frame: int | None = None
    ref_d_eq_volume_px: float | None = None
    ref_d_eq_volume_mm: float | None = None
    ref_d_eq_area_px: float | None = None
    ref_d_eq_area_mm: float | None = None
    ref_d_parallel_px: float | None = None
    ref_d_parallel_mm: float | None = None
    ref_d_normal_px: float | None = None
    ref_d_normal_mm: float | None = None
    ref_clearance_px: float | None = None
    ref_clearance_mm: float | None = None


@dataclass(frozen=True)
class ComponentMeasurement:
    d_spread_px: float
    area_px2: int
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray
    contact_bbox: tuple[int, int, int, int] | None = None
    rim_bbox: tuple[int, int, int, int] | None = None
    d_contact_px: float | None = None
    d_rim_px: float | None = None
    d_parallel_px: float | None = None
    d_normal_px: float | None = None
    d_eq_volume_px: float | None = None
    d_eq_area_px: float | None = None
    surface_clearance_px: float | None = None
    surface_parallel_line: tuple[int, int, int, int] | None = None
    surface_normal_line: tuple[int, int, int, int] | None = None


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
