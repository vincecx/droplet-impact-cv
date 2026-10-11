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
                "frame_number",
                "time_ms",
                "d_spread_px",
                "d_spread_mm",
                "d_contact_px",
                "d_contact_mm",
                "d_rim_px",
                "d_rim_mm",
                "ref_frame",
                "ref_d_eq_volume_px",
                "ref_d_eq_volume_mm",
                "ref_d_eq_area_px",
                "ref_d_eq_area_mm",
                "ref_d_parallel_px",
                "ref_d_parallel_mm",
                "ref_d_normal_px",
                "ref_d_normal_mm",
                "ref_clearance_px",
                "ref_clearance_mm",
                "impact_frame",
                "surface_y_px",
                "droplet_area_px2",
                "fps",
                "pixel_size_mm_per_px",
                "calibration_frame",
            ]
        )
        for row in measurements:
            writer.writerow(
                [
                    row.filename,
                    row.frame_number,
                    f"{row.time_ms:.6f}",
                    _format_required(row.d_spread_px, 3),
                    _format_required(row.d_spread_mm, 9),
                    _format_optional(row.d_contact_px, 3),
                    _format_optional(row.d_contact_mm, 9),
                    _format_optional(row.d_rim_px, 3),
                    _format_optional(row.d_rim_mm, 9),
                    "" if row.ref_frame is None else row.ref_frame,
                    _format_optional(row.ref_d_eq_volume_px, 3),
                    _format_optional(row.ref_d_eq_volume_mm, 9),
                    _format_optional(row.ref_d_eq_area_px, 3),
                    _format_optional(row.ref_d_eq_area_mm, 9),
                    _format_optional(row.ref_d_parallel_px, 3),
                    _format_optional(row.ref_d_parallel_mm, 9),
                    _format_optional(row.ref_d_normal_px, 3),
                    _format_optional(row.ref_d_normal_mm, 9),
                    _format_optional(row.ref_clearance_px, 3),
                    _format_optional(row.ref_clearance_mm, 9),
                    "" if row.impact_frame is None else row.impact_frame,
                    row.surface_y_px,
                    row.droplet_area_px2,
                    row.fps,
                    row.pixel_size_mm_per_px,
                    "" if row.calibration_frame is None else row.calibration_frame,
                ]
            )


def _format_optional(value: float | None, precision: int) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{precision}f}"


def _format_required(value: float, precision: int) -> str:
    return "" if math.isnan(value) else f"{value:.{precision}f}"
