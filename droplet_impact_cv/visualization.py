from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import SurfaceLine


DEBUG_MASK_ALPHA = 0.2
CONTACT_LINE_COLOR = (0, 255, 0)
RIM_TO_RIM_COLOR = (255, 0, 255)
SURFACE_PARALLEL_COLOR = (255, 128, 0)
SURFACE_NORMAL_COLOR = (0, 128, 255)
DEBUG_FONT = cv2.FONT_HERSHEY_DUPLEX
DEBUG_FONT_SCALE = 0.7


def clear_debug_overlays(debug_dir: Path) -> None:
    """Delete frame-numbered overlay PNGs so the directory holds one run only.

    Overlays are written selectively (every N frames plus around impact), so
    the directory must be cleared first to keep every remaining overlay
    consistent with the current run's calibration.
    """
    if not debug_dir.is_dir():
        return
    for path in debug_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"):
        path.unlink()


def write_debug_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray | None,
    surface_line: SurfaceLine,
    bbox: tuple[int, int, int, int] | None,
    frame_number: int,
    spreading_diameter_mm: float,
    rim_bbox: tuple[int, int, int, int] | None = None,
    contact_line_diameter_mm: float | None = None,
    rim_to_rim_diameter_mm: float | None = None,
    surface_parallel_line: tuple[int, int, int, int] | None = None,
    surface_normal_line: tuple[int, int, int, int] | None = None,
    initial_equivalent_diameter_mm: float | None = None,
    initial_surface_parallel_diameter_mm: float | None = None,
    initial_surface_normal_diameter_mm: float | None = None,
) -> None:
    lo, hi = np.percentile(image, [0.5, 99.5])
    gray = np.clip((image - lo) / max(1.0, hi - lo) * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        liquid_pixels = mask > 0
        overlay[liquid_pixels] = (
            overlay[liquid_pixels].astype(np.float32) * (1.0 - DEBUG_MASK_ALPHA)
            + np.array([0, 0, 255], dtype=np.float32) * DEBUG_MASK_ALPHA
        ).astype(np.uint8)
    width = overlay.shape[1]
    surface_start = (0, int(round(surface_line.y_at(0, width))))
    surface_end = (width - 1, int(round(surface_line.y_at(width - 1, width))))
    cv2.line(overlay, surface_start, surface_end, (0, 255, 255), 1)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.line(overlay, (x0, y0), (x1, y1), CONTACT_LINE_COLOR, 2)
    if rim_bbox is not None:
        x0, y0, x1, y1 = rim_bbox
        cv2.line(overlay, (x0, y0), (x1, y1), RIM_TO_RIM_COLOR, 2)
        rim_label_mm = (
            rim_to_rim_diameter_mm
            if rim_to_rim_diameter_mm is not None
            else spreading_diameter_mm
        )
        cv2.putText(
            overlay,
            f"rim-to-rim={rim_label_mm:.4f} mm",
            (12, 80 if surface_parallel_line is not None else 54),
            DEBUG_FONT,
            DEBUG_FONT_SCALE,
            RIM_TO_RIM_COLOR,
            1,
            cv2.LINE_AA,
        )
    if surface_parallel_line is not None:
        x0, y0, x1, y1 = surface_parallel_line
        cv2.line(overlay, (x0, y0), (x1, y1), SURFACE_PARALLEL_COLOR, 2)
    if surface_normal_line is not None:
        x0, y0, x1, y1 = surface_normal_line
        cv2.line(overlay, (x0, y0), (x1, y1), SURFACE_NORMAL_COLOR, 2)
    if initial_equivalent_diameter_mm is not None:
        text = f"frame={frame_number} D0_eq={initial_equivalent_diameter_mm:.4f} mm"
    elif rim_bbox is not None and contact_line_diameter_mm is not None:
        text = f"frame={frame_number} contact={contact_line_diameter_mm:.4f} mm"
    else:
        text = f"frame={frame_number} spreading={spreading_diameter_mm:.4f} mm"
    cv2.putText(
        overlay,
        text,
        (12, 28),
        DEBUG_FONT,
        DEBUG_FONT_SCALE,
        CONTACT_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    if (
        initial_surface_parallel_diameter_mm is not None
        and initial_surface_normal_diameter_mm is not None
    ):
        cv2.putText(
            overlay,
            (
                f"D0_parallel={initial_surface_parallel_diameter_mm:.4f} mm "
                f"D0_normal={initial_surface_normal_diameter_mm:.4f} mm"
            ),
            (12, 54),
            DEBUG_FONT,
            DEBUG_FONT_SCALE,
            SURFACE_PARALLEL_COLOR,
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
