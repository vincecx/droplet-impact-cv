from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import SurfaceLine


DEBUG_MASK_ALPHA = 0.2
CONTACT_LINE_COLOR = (0, 255, 0)
RIM_TO_RIM_COLOR = (255, 0, 255)
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


def render_debug_overlay(
    image: np.ndarray,
    mask: np.ndarray | None,
    surface_line: SurfaceLine,
    bbox: tuple[int, int, int, int] | None,
    frame_number: int,
    d_spread_mm: float,
    rim_bbox: tuple[int, int, int, int] | None = None,
    d_contact_mm: float | None = None,
    d_rim_mm: float | None = None,
) -> np.ndarray:
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
    if d_contact_mm is not None:
        cv2.putText(
            overlay,
            f"d_contact={d_contact_mm:.4f} mm",
            (12, 28),
            DEBUG_FONT,
            DEBUG_FONT_SCALE,
            CONTACT_LINE_COLOR,
            1,
            cv2.LINE_AA,
        )
    if rim_bbox is not None:
        x0, y0, x1, y1 = rim_bbox
        cv2.line(overlay, (x0, y0), (x1, y1), RIM_TO_RIM_COLOR, 2)
        rim_label_mm = (
            d_rim_mm
            if d_rim_mm is not None
            else d_spread_mm
        )
        cv2.putText(
            overlay,
            f"d_rim={rim_label_mm:.4f} mm",
            (12, 54),
            DEBUG_FONT,
            DEBUG_FONT_SCALE,
            RIM_TO_RIM_COLOR,
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_debug_overlay(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray | None,
    surface_line: SurfaceLine,
    bbox: tuple[int, int, int, int] | None,
    frame_number: int,
    d_spread_mm: float,
    rim_bbox: tuple[int, int, int, int] | None = None,
    d_contact_mm: float | None = None,
    d_rim_mm: float | None = None,
) -> None:
    overlay = render_debug_overlay(
        image, mask, surface_line, bbox, frame_number, d_spread_mm,
        rim_bbox, d_contact_mm, d_rim_mm,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
