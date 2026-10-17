"""Conservative shadow rejection as a tracked drop approaches the surface.

The ellipse supports a contour envelope, never a replacement silhouette. Only
the lower fringe is bounded, using edge thickness measured on the same drop's
unobstructed upper contour. Sharp sheets and deformed drops retain faint growth.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from .models import SurfaceLine


DETACHED_ELLIPSE_MIN_IOU = 0.96
CONTACT_ELLIPSE_MIN_IOU = 0.94
MAX_CONTACT_TRUNCATION = 0.05
MIN_DIFFUSE_EDGE_SCALE_PX = 4.0


@dataclass(frozen=True)
class FreeDropModel:
    frame_number: int
    center: tuple[float, float]
    axes: tuple[float, float]
    bbox: tuple[int, int, int, int]
    clearance_px: float
    surface_slope: float = 0.0
    angle_deg: float = 0.0
    is_detached: bool = True


def fit_free_drop_model(
    core: np.ndarray, surface: SurfaceLine, top_rows: int, band_px: int,
    frame_number: int, *, allow_contact: bool = False,
) -> FreeDropModel | None:
    """Fit a complete ellipse, or a slightly truncated initial-contact ellipse."""
    ys, xs = np.nonzero(core)
    if xs.size < 25:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    if x0 == 0 or x1 == core.shape[1] - 1 or y0 <= top_rows or y1 == core.shape[0] - 1:
        return None
    clearance = float(np.min((surface.y_at(xs, core.shape[1]) - ys)
                             / np.hypot(1, surface.slope)))
    detached = clearance > band_px + 1
    if not detached and not allow_contact:
        return None
    contours, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 12:
        return None
    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    center, axes, _ = ellipse
    if not np.isfinite((*center, *axes, ellipse[2])).all() or min(axes) < 6 or max(axes) > 3 * min(axes):
        return None
    fitted = np.zeros(core.shape, dtype=np.uint8)
    cv2.ellipse(fitted, ellipse, 1, -1)
    fitted = fitted > 0
    overlap = np.count_nonzero(fitted & core) / max(1, np.count_nonzero(fitted | core))
    if overlap < (DETACHED_ELLIPSE_MIN_IOU if detached else CONTACT_ELLIPSE_MIN_IOU):
        return None
    if not detached:
        yy, xx = np.nonzero(fitted)
        below_surface = yy > surface.y_at(xx, core.shape[1])
        # Only a small truncation at initial contact is supported by the history.
        if np.mean(below_surface) > MAX_CONTACT_TRUNCATION:
            return None
    return FreeDropModel(frame_number, tuple(center), tuple(axes),
                         (x0, y0, x1, y1), clearance, surface.slope,
                         float(ellipse[2]), detached)


def has_consistent_history(current: FreeDropModel, previous: tuple[FreeDropModel, ...]) -> bool:
    """Check size and extrapolated position using actual filename intervals."""
    if len(previous) != 2:
        return False
    first, second = previous
    if not first.is_detached or not second.is_detached:
        return False
    dt = second.frame_number - first.frame_number
    next_dt = current.frame_number - second.frame_number
    if dt <= 0 or next_dt <= 0:
        return False
    for earlier in previous:
        if np.max(np.abs(np.sort(current.axes) / np.sort(earlier.axes) - 1)) > 0.15:
            return False
    velocity = (np.asarray(second.center) - first.center) / dt
    predicted = np.asarray(second.center) + velocity * next_dt
    tolerance = max(3.0, 0.1 * min(current.axes))
    return float(np.linalg.norm(np.asarray(current.center) - predicted)) <= tolerance


def constrain_shadow_fringe(
    candidate: np.ndarray, core: np.ndarray, model: FreeDropModel,
    previous: tuple[FreeDropModel, ...], band_px: int,
    image: np.ndarray, background: np.ndarray, weak_change_threshold: float,
    surface: SurfaceLine,
) -> tuple[np.ndarray, int]:
    """Trim an anomalous lower fringe before or at initial surface contact.

    A full-image gradient threshold would erase transparent liquid. Instead,
    estimate the normal edge-transition thickness from the upper 70% of the
    observed candidate contour and apply that envelope only to the lower 30%.
    Detached drops retain all strong pixels. At initial contact, require a
    nearly complete ellipse and diffuse outer edges before using its envelope.
    """
    if not has_consistent_history(model, previous):
        return candidate, 0
    _, y0, _, y1 = model.bbox
    if model.clearance_px > (y1 - y0 + 1) / 2:
        return candidate, 0
    labels, _ = ndi.label(candidate)
    labels_in_core = labels[core]
    labels_in_core = labels_in_core[labels_in_core > 0]
    if not labels_in_core.size:
        return candidate, 0
    label = int(np.argmax(np.bincount(labels_in_core)))
    component = labels == label
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    core_ys, core_xs = np.nonzero(core)
    core_normal = core_ys - model.surface_slope * core_xs
    lower_start = float(core_normal.min() + 0.7 * np.ptp(core_normal))
    upper_points = points[points[:, 1] - model.surface_slope * points[:, 0] < lower_start]
    if len(upper_points) < 12:
        return candidate, 0
    distance = ndi.distance_transform_edt(~core)
    thickness = float(np.percentile(distance[upper_points[:, 1], upper_points[:, 0]], 90)) + 1.0
    if thickness > 0.1 * min(model.axes):
        return candidate, 0
    if model.is_detached and model.clearance_px <= thickness + band_px + 1:
        return candidate, 0
    normal = (np.arange(candidate.shape[0])[:, None]
              - model.surface_slope * np.arange(candidate.shape[1])[None, :])
    lower = normal >= lower_start
    if not model.is_detached:
        envelope = np.zeros(core.shape, dtype=np.uint8)
        cv2.ellipse(envelope, (model.center, model.axes, model.angle_deg), 1, -1)
        distance = ndi.distance_transform_edt(envelope == 0)
        boundary = component & ~ndi.binary_erosion(component)
        yy, xx = np.indices(core.shape)
        gap = surface.y_at(xx, core.shape[1]) - yy
        outer_edge = (boundary & (distance > thickness) & (gap >= band_px + 1)
                      & (gap <= max(4, 0.1 * max(model.axes))))
        if np.count_nonzero(outer_edge) < max(4, int(0.08 * min(model.axes))):
            return candidate, 0
        change = ndi.gaussian_filter(background - image, 1.0)
        gradient = np.hypot(ndi.sobel(change, 0) / 8, ndi.sobel(change, 1) / 8)
        sharpness = gradient / (np.abs(change) + weak_change_threshold)
        # A normalized transition scale under four pixels is evidence of a real
        # thin sheet. Decline shape correction rather than erasing that edge.
        if float(np.median(sharpness[outer_edge])) >= 1.0 / MIN_DIFFUSE_EDGE_SCALE_PX:
            return candidate, 0
    rejected = component & lower & (distance > thickness)
    return candidate & ~rejected, int(np.count_nonzero(rejected))
