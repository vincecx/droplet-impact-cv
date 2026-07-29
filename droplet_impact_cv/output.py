from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from .models import FrameMeasurement


def write_csv(path: Path, measurements: Iterable[FrameMeasurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "frame",
                "time_ms",
                "spreading_diameter_px",
                "spreading_diameter_mm",
                "contact_line_diameter_px",
                "contact_line_diameter_mm",
                "rim_to_rim_diameter_px",
                "rim_to_rim_diameter_mm",
                "reference_frame",
                "initial_equivalent_diameter_px",
                "initial_equivalent_diameter_mm",
                "initial_surface_parallel_diameter_px",
                "initial_surface_parallel_diameter_mm",
                "initial_surface_normal_diameter_px",
                "initial_surface_normal_diameter_mm",
                "surface_clearance_px",
                "surface_clearance_mm",
                "impact_frame",
                "surface_y_px",
                "component_area_px",
                "fps",
                "pixel-size-mm",
                "surface-frame",
            ]
        )
        for row in measurements:
            writer.writerow(
                [
                    row.filename,
                    row.frame_number,
                    f"{row.time_ms:.6f}",
                    _format_required(row.spreading_diameter_px, 3),
                    _format_required(row.spreading_diameter_mm, 9),
                    _format_optional(row.contact_line_diameter_px, 3),
                    _format_optional(row.contact_line_diameter_mm, 9),
                    _format_optional(row.rim_to_rim_diameter_px, 3),
                    _format_optional(row.rim_to_rim_diameter_mm, 9),
                    "" if row.reference_frame is None else row.reference_frame,
                    _format_optional(row.initial_equivalent_diameter_px, 3),
                    _format_optional(row.initial_equivalent_diameter_mm, 9),
                    _format_optional(row.initial_surface_parallel_diameter_px, 3),
                    _format_optional(row.initial_surface_parallel_diameter_mm, 9),
                    _format_optional(row.initial_surface_normal_diameter_px, 3),
                    _format_optional(row.initial_surface_normal_diameter_mm, 9),
                    _format_optional(row.surface_clearance_px, 3),
                    _format_optional(row.surface_clearance_mm, 9),
                    "" if row.impact_frame is None else row.impact_frame,
                    row.surface_y,
                    row.component_area_px,
                    row.fps,
                    row.pixel_size_mm,
                    "" if row.surface_frame is None else row.surface_frame,
                ]
            )


def _format_optional(value: float | None, precision: int) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{precision}f}"


def _format_required(value: float, precision: int) -> str:
    return "" if math.isnan(value) else f"{value:.{precision}f}"
