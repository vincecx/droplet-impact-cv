from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .models import (
    DEFAULT_SURFACE_BAND_PX,
    DEFAULT_WEAK_REACH_PX,
    AnalysisConfig,
    ComponentMeasurement,
    SurfaceLine,
)


SURFACE_MASK_BELOW_TOLERANCE_PX = 0
DEFAULT_CALIBRATION_EDGE_GAP_PX = 20
FRAME_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{6})$")
SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".jpg", ".jpeg")
MAX_REFLECTION_SYMMETRY_ERROR = 0.1


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def frame_number_from_filename(path: Path) -> int:
    match = FRAME_NUMBER_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"TIFF filename must end with a six-digit frame number: {path.name}"
        )
    return int(match.group(1))


def find_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No TIFF or JPEG images found in: {input_dir}")

    numbered_files = sorted(
        ((frame_number_from_filename(path), path) for path in files),
        key=lambda item: (item[0], natural_sort_key(item[1])),
    )
    for (frame_number, first), (next_frame_number, second) in zip(
        numbered_files, numbered_files[1:]
    ):
        if frame_number == next_frame_number:
            raise ValueError(
                f"Duplicate frame number {frame_number:06d}: {first.name}, {second.name}"
            )
    return [path for _, path in numbered_files]


def find_tiff_files(input_dir: Path) -> list[Path]:
    """Backward-compatible alias for callers using the original API name."""
    return find_image_files(input_dir)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Existing thresholds are expressed in the camera's 12-bit gray-level
        # scale. Preserve that scale when 8-bit JPEG exports are supplied.
        if image.dtype == np.uint8:
            image = image.astype(np.float32) * (4095.0 / 255.0)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}: {path}")
    return image.astype(np.float32, copy=False)


def build_background(files: list[Path]) -> np.ndarray:
    if not files:
        raise ValueError("At least one TIFF file is required to build the background")
    background_file = min(files, key=frame_number_from_filename)
    return read_image(background_file)


def estimate_surface_y(background: np.ndarray, search_start_px: int, drop_delta: float) -> int:
    row_profile = np.median(background, axis=1)
    smooth = ndi.gaussian_filter1d(row_profile, sigma=3.0)
    bright_reference = float(np.percentile(smooth[: max(20, search_start_px)], 95))
    limit = bright_reference - drop_delta

    for y in range(max(0, search_start_px), len(smooth) - 20):
        window = smooth[y : y + 20]
        if np.mean(window < limit) >= 0.8:
            return y

    gradient = np.abs(np.diff(smooth))
    return int(np.argmax(gradient[search_start_px:]) + search_start_px)


def estimate_threshold(
    files: list[Path],
    background: np.ndarray,
    surface_y_px: int,
    sample_frame_count: int,
    min_strong_threshold: float,
) -> float:
    sample_files = files[: min(sample_frame_count, len(files))]
    if len(sample_files) < 2:
        return min_strong_threshold

    roi_bottom = max(10, min(surface_y_px - 30, background.shape[0]))
    noise_samples = []
    for path in sample_files:
        image = read_image(path)
        diff = np.abs(background[:roi_bottom] - image[:roi_bottom])
        noise_samples.append(diff.ravel())

    noise = np.concatenate(noise_samples)
    median = float(np.median(noise))
    mad = float(np.median(np.abs(noise - median)))
    sigma = 1.4826 * mad
    return max(min_strong_threshold, median + 8.0 * sigma)


def make_structure(radius_px: int) -> np.ndarray:
    radius_px = max(1, int(radius_px))
    size = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean_binary_mask(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, structure) > 0
    opened = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_OPEN, structure) > 0

    # Use the opened regions as reliable foreground markers, then reconstruct
    # their original closed shapes.  A conventional opening removes narrow
    # but real contour sections, such as a dark contact-line boundary beside
    # a bright reflection.  Reconstruction restores such sections when they
    # remain connected to a robust liquid region while still rejecting small,
    # disconnected foreground artifacts that the opening removed completely.
    reconstructed = ndi.binary_propagation(opened, mask=closed)
    return ndi.binary_fill_holes(reconstructed).astype(np.uint8)


def hysteresis_change_mask(
    image: np.ndarray,
    background: np.ndarray,
    threshold: float,
    weak_threshold: float | None,
    roi: np.ndarray | None = None,
    max_propagation_px: int = DEFAULT_WEAK_REACH_PX,
) -> np.ndarray:
    """Detect changed pixels with an optional weak-threshold extension.

    Transparent liquid can appear brighter than the local background, for
    example a glassy lamella refracting the backlight over a dark substrate,
    so both darkening and brightening count as foreground.  A fully
    transparent film transmits the backlight almost unchanged, leaving only
    part of its silhouette above the strong threshold.  Pixels above the weak
    threshold are kept when they connect to a strong detection, which extends
    the confident regions across the faint interior without admitting
    isolated weak artifacts such as diffuse shadows.
    """
    change = np.abs(background - image)
    strong = change >= threshold
    if roi is not None:
        strong &= roi
    if weak_threshold is None or weak_threshold >= threshold:
        return strong
    weak = change >= weak_threshold
    if roi is not None:
        weak &= roi
    # Bound the growth distance so a diffuse out-of-focus artifact that
    # touches the droplet cannot pull the whole artifact into the mask.
    # Transparent-film interiors sit well within this distance of a strong
    # rim or highlight detection.
    return ndi.binary_dilation(strong, mask=weak, iterations=max_propagation_px)


def foreground_mask(
    image: np.ndarray,
    background: np.ndarray,
    surface_line: SurfaceLine,
    threshold: float,
    config: AnalysisConfig,
    structure: np.ndarray,
    weak_threshold: float | None = None,
) -> np.ndarray:
    yy, xx = np.indices(image.shape)
    distance_from_surface = yy - surface_line.y_at(xx, image.shape[1])
    roi = distance_from_surface <= config.foreground_depth_px
    changed = hysteresis_change_mask(
        image,
        background,
        threshold,
        weak_threshold,
        roi,
        config.weak_reach_px,
    )
    return clean_binary_mask(changed.astype(np.uint8), structure)


def full_foreground_mask(
    image: np.ndarray,
    background: np.ndarray,
    threshold: float,
    structure: np.ndarray,
    weak_threshold: float | None = None,
    weak_reach_px: int = DEFAULT_WEAK_REACH_PX,
) -> np.ndarray:
    changed = hysteresis_change_mask(
        image,
        background,
        threshold,
        weak_threshold,
        max_propagation_px=weak_reach_px,
    )
    return clean_binary_mask(changed, structure)


def largest_component(mask: np.ndarray, min_droplet_area_px2: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)

    best_label = 0
    best_area = 0
    for label in range(1, count + 1):
        area = int(np.sum(labels == label))
        if area >= min_droplet_area_px2 and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return np.zeros(mask.shape, dtype=bool)
    return labels == best_label


def component_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def select_liquid_component(
    mask: np.ndarray,
    surface_line: SurfaceLine,
    config: AnalysisConfig,
    preferred_center_x: float | None = None,
) -> tuple[np.ndarray, int]:
    labels, count = ndi.label(mask)
    empty = np.zeros(mask.shape, dtype=bool)
    if count == 0:
        return empty, 0

    yy, xx = np.indices(mask.shape)
    distance_from_surface = yy - surface_line.y_at(xx, mask.shape[1])
    # A reference droplet is measured while it is still separated from the
    # substrate, so candidate selection must not require a surface contact.
    # Restrict candidates to the gas-side silhouette and prefer the one closest
    # to the surface; this rejects the mirror image below the substrate.
    gas_side = distance_from_surface <= 0

    best_label = 0
    best_clearance = math.inf
    best_area = 0
    best_center_distance = math.inf

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_droplet_area_px2:
            continue
        gas_side_pixels = int((component & gas_side).sum())
        if gas_side_pixels < config.min_contact_px:
            continue
        bbox = component_bbox(component)
        assert bbox is not None
        x_min, _y_min, x_max, _y_max = bbox
        center_distance = (
            abs(((x_min + x_max) / 2.0) - preferred_center_x)
            if preferred_center_x is not None
            else 0.0
        )
        bottom_clearance_px = max(0.0, -float(distance_from_surface[component].max()))
        candidate_score = (center_distance, bottom_clearance_px, -area)
        best_score = (best_center_distance, best_clearance, -best_area)
        if candidate_score < best_score:
            best_label = label
            best_center_distance = center_distance
            best_clearance = bottom_clearance_px
            best_area = area

    if best_label == 0:
        return empty, 0
    return labels == best_label, best_area


def longest_true_run(active: np.ndarray, close_size: int = 5) -> tuple[int, int] | None:
    active = np.asarray(active, dtype=bool)
    if close_size > 1:
        active = ndi.binary_closing(active, structure=np.ones(close_size, dtype=bool))
    if not active.any():
        return None

    labels, count = ndi.label(active)
    best_label = 0
    best_width = 0
    for label in range(1, count + 1):
        xs = np.flatnonzero(labels == label)
        width = int(xs[-1] - xs[0] + 1)
        if width > best_width:
            best_label = label
            best_width = width

    xs = np.flatnonzero(labels == best_label)
    return int(xs[0]), int(xs[-1])


def surface_band_mask(
    shape: tuple[int, int],
    surface_line: SurfaceLine,
    below_tolerance_px: int,
) -> np.ndarray:
    yy, xx = np.indices(shape)
    distance_from_surface = yy - surface_line.y_at(xx, shape[1])
    return distance_from_surface <= below_tolerance_px


def spreading_profile_on_surface(
    component: np.ndarray,
    surface_line: SurfaceLine,
    half_width_px: int = 1,
) -> np.ndarray:
    height, width = component.shape
    xs = np.arange(width)
    ys = np.rint(surface_line.y_at(xs, width)).astype(int)
    profile = np.zeros(width, dtype=bool)
    for offset in range(-half_width_px, half_width_px + 1):
        sample_y = ys + offset
        valid = (sample_y >= 0) & (sample_y < height)
        profile[valid] |= component[sample_y[valid], xs[valid]]
    return profile


def surface_intersection_bounds(
    component: np.ndarray,
    surface_line: SurfaceLine,
    band_px: int = DEFAULT_SURFACE_BAND_PX,
) -> tuple[int, int] | None:
    """Outermost x bounds where the silhouette meets the surface-line band.

    The band spans ``band_px`` above and below the line, absorbing the
    pixel-level wobble of a blurred contact zone without measuring liquid
    that is clearly detached from the surface.
    """
    yy, xx = np.indices(component.shape)
    distance = yy - surface_line.y_at(xx, component.shape[1])
    band = np.abs(distance) <= band_px
    profile = np.any(component & band, axis=0)
    xs = np.flatnonzero(profile)
    if xs.size == 0:
        return None
    return int(xs[0]), int(xs[-1])


def rim_to_rim_segment(
    component: np.ndarray,
    surface_line: SurfaceLine,
) -> tuple[float, tuple[int, int, int, int]] | None:
    """Find the longest droplet chord parallel to the surface line.

    Each foreground pixel is projected onto axes parallel and perpendicular to
    the surface. Pixels within the same one-pixel perpendicular band form one
    candidate chord. The longest candidate defines the hydrophobic
    rim-to-rim diameter and its displayed measurement line.
    """
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None

    slope = surface_line.slope
    length_scale = math.sqrt(1.0 + slope**2)
    parallel_x = 1.0 / length_scale
    parallel_y = slope / length_scale
    normal_x = -slope / length_scale
    normal_y = 1.0 / length_scale
    parallel = xs * parallel_x + ys * parallel_y
    normal = xs * normal_x + ys * normal_y
    normal_bands = np.rint(normal).astype(int)

    best_length = -1.0
    best_parallel_min = 0.0
    best_parallel_max = 0.0
    best_normal = 0.0
    for normal_band in np.unique(normal_bands):
        in_band = normal_bands == normal_band
        band_parallel = parallel[in_band]
        candidate_min = float(band_parallel.min())
        candidate_max = float(band_parallel.max())
        candidate_length = candidate_max - candidate_min + 1.0
        if candidate_length > best_length:
            best_length = candidate_length
            best_parallel_min = candidate_min
            best_parallel_max = candidate_max
            best_normal = float(normal[in_band].mean())

    def point_at(parallel_position: float) -> tuple[int, int]:
        x = parallel_position * parallel_x + best_normal * normal_x
        y = parallel_position * parallel_y + best_normal * normal_y
        return int(round(x)), int(round(y))

    x_min, y_min = point_at(best_parallel_min)
    x_max, y_max = point_at(best_parallel_max)
    return best_length, (x_min, y_min, x_max, y_max)


def surface_aligned_dimensions(
    component: np.ndarray,
    surface_line: SurfaceLine,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
] | None:
    """Measure a silhouette in axes tangent and normal to the substrate.

    The surface-parallel and surface-normal diameters are full projection
    spans. Their geometric mean ``(D_parallel² D_normal)^(1/3)`` is the
    equivalent spherical diameter of the axisymmetric approximation.
    """
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None

    slope = surface_line.slope
    scale = math.sqrt(1.0 + slope**2)
    tangent_x, tangent_y = 1.0 / scale, slope / scale
    normal_x, normal_y = -slope / scale, 1.0 / scale
    tangent = xs * tangent_x + ys * tangent_y
    normal = xs * normal_x + ys * normal_y
    tangent_min, tangent_max = float(tangent.min()), float(tangent.max())
    normal_min, normal_max = float(normal.min()), float(normal.max())
    d_parallel_px = tangent_max - tangent_min + 1.0
    d_normal_px = normal_max - normal_min + 1.0
    d_eq_volume_px = (
        d_normal_px * d_parallel_px**2
    ) ** (1.0 / 3.0)

    def point_at(tangent_position: float, normal_position: float) -> tuple[int, int]:
        x = tangent_position * tangent_x + normal_position * normal_x
        y = tangent_position * tangent_y + normal_position * normal_y
        return int(round(x)), int(round(y))

    parallel_line = (
        *point_at(tangent_min, (normal_min + normal_max) / 2.0),
        *point_at(tangent_max, (normal_min + normal_max) / 2.0),
    )
    normal_line = (
        *point_at((tangent_min + tangent_max) / 2.0, normal_min),
        *point_at((tangent_min + tangent_max) / 2.0, normal_max),
    )
    signed_normal_distance = (ys - surface_line.y_at(xs, component.shape[1])) / scale
    surface_clearance_px = max(0.0, -float(signed_normal_distance.max()))
    return (
        d_parallel_px,
        d_normal_px,
        d_eq_volume_px,
        surface_clearance_px,
        parallel_line,
        normal_line,
    )


def fill_droplet_interior(
    component: np.ndarray,
    surface_line: SurfaceLine,
    seal_half_width_px: int = 1,
) -> np.ndarray:
    """Seal the substrate contact span and fill the droplet interior.

    An internal bright region can leak into the background through the contact
    line and survive a conventional hole fill.  By default, the liquid contact
    span is continuous and the selected droplet contains no internal air gaps.
    Top- and side-open exterior concavities remain unchanged.
    """
    filled = ndi.binary_fill_holes(np.asarray(component, dtype=bool))
    profile = spreading_profile_on_surface(filled, surface_line, seal_half_width_px)
    contact_xs = np.flatnonzero(profile)
    if contact_xs.size == 0:
        return filled

    height, width = filled.shape
    xs = np.arange(int(contact_xs[0]), int(contact_xs[-1]) + 1)
    surface_ys = np.rint(surface_line.y_at(xs, width)).astype(int)
    for offset in range(-seal_half_width_px, seal_half_width_px + 1):
        ys = surface_ys + offset
        valid = (ys >= 0) & (ys < height)
        filled[ys[valid], xs[valid]] = True

    return ndi.binary_fill_holes(filled)


def component_measurement(
    mask: np.ndarray,
    surface_line: SurfaceLine,
    config: AnalysisConfig,
    preferred_center_x: float | None = None,
) -> ComponentMeasurement:
    component, area = select_liquid_component(
        mask, surface_line, config, preferred_center_x
    )
    if area == 0:
        return ComponentMeasurement(0.0, 0, None, component)

    # Seal apparent holes that open through a few pixels near the substrate,
    # then treat the selected droplet's enclosed bright regions as liquid.
    component = fill_droplet_interior(component, surface_line)
    aligned_dimensions = surface_aligned_dimensions(component, surface_line)
    # Circle with the same filled silhouette area; measure before surface clipping,
    # as for the volume-equivalent diameter and the two projection spans.
    d_eq_area_px = math.sqrt(4.0 * int(component.sum()) / math.pi)

    # Hydrophobic droplets are measured rim-to-rim. Otherwise, the diameter is
    # the silhouette width where it meets the fixed surface-line band. Both
    # measurements are taken before clipping, so pixels just below the line
    # still participate in the contact-line measurement.
    contact_bounds = surface_intersection_bounds(
        component, surface_line, config.surface_band_px
    )
    contact_bbox = None
    d_contact_px = 0.0
    if contact_bounds is not None:
        contact_x_min, contact_x_max = contact_bounds
        contact_bbox = (
            contact_x_min,
            int(round(surface_line.y_at(contact_x_min, component.shape[1]))),
            contact_x_max,
            int(round(surface_line.y_at(contact_x_max, component.shape[1]))),
        )
        d_contact_px = float(
            (contact_x_max - contact_x_min + 1)
            * math.sqrt(1.0 + surface_line.slope**2)
        )
    if config.hydrophobic:
        rim_measurement = rim_to_rim_segment(component, surface_line)
    else:
        rim_measurement = None

    # The reported mask and area cover only the liquid on and above the line.
    component &= surface_band_mask(
        component.shape,
        surface_line,
        SURFACE_MASK_BELOW_TOLERANCE_PX,
    )
    clipped_component = largest_component(component, config.min_droplet_area_px2)
    if clipped_component.any():
        component = clipped_component
    area = int(component.sum())
    if aligned_dimensions is None:
        d_parallel_px = None
        d_normal_px = None
        d_eq_volume_px = None
        surface_clearance_px = None
        surface_parallel_line = None
        surface_normal_line = None
    else:
        (
            d_parallel_px,
            d_normal_px,
            d_eq_volume_px,
            surface_clearance_px,
            surface_parallel_line,
            surface_normal_line,
        ) = aligned_dimensions
    if config.hydrophobic:
        if rim_measurement is None:
            return ComponentMeasurement(0.0, area, None, component)
        d_spread_px, rim_bbox = rim_measurement
        return ComponentMeasurement(
            d_spread_px,
            area,
            rim_bbox,
            component,
            contact_bbox=contact_bbox,
            rim_bbox=rim_bbox,
            d_contact_px=d_contact_px,
            d_rim_px=d_spread_px,
            d_parallel_px=d_parallel_px,
            d_normal_px=d_normal_px,
            d_eq_volume_px=d_eq_volume_px,
            d_eq_area_px=d_eq_area_px,
            surface_clearance_px=surface_clearance_px,
            surface_parallel_line=surface_parallel_line,
            surface_normal_line=surface_normal_line,
        )

    if contact_bounds is None:
        assert aligned_dimensions is not None
        (
            d_parallel_px,
            d_normal_px,
            d_eq_volume_px,
            surface_clearance_px,
            surface_parallel_line,
            surface_normal_line,
        ) = aligned_dimensions
        return ComponentMeasurement(
            0.0,
            area,
            None,
            component,
            d_parallel_px=d_parallel_px,
            d_normal_px=d_normal_px,
            d_eq_volume_px=d_eq_volume_px,
            d_eq_area_px=d_eq_area_px,
            surface_clearance_px=surface_clearance_px,
            surface_parallel_line=surface_parallel_line,
            surface_normal_line=surface_normal_line,
        )
    assert contact_bbox is not None
    x_min, y_min, x_max, y_max = contact_bbox
    d_spread_px = d_contact_px
    return ComponentMeasurement(
        d_spread_px,
        area,
        (x_min, y_min, x_max, y_max),
        component,
        contact_bbox=contact_bbox,
        d_contact_px=d_spread_px,
        d_parallel_px=d_parallel_px,
        d_normal_px=d_normal_px,
        d_eq_volume_px=d_eq_volume_px,
        d_eq_area_px=d_eq_area_px,
        surface_clearance_px=surface_clearance_px,
        surface_parallel_line=surface_parallel_line,
        surface_normal_line=surface_normal_line,
    )


def touches_surface(measurement: ComponentMeasurement, config: AnalysisConfig) -> bool:
    """Return whether the contact-line diameter marks the start of impact."""
    return (
        measurement.d_contact_px is not None
        and measurement.d_contact_px >= config.min_impact_width_px
    )


def select_calibration_component(
    mask: np.ndarray,
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    labels, count = ndi.label(mask)
    best_label = 0
    best_score = -1

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_droplet_area_px2:
            continue

        bbox = component_bbox(component)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        vertical_span = y_max - y_min + 1
        below_surface = int(component[coarse_surface_y:].sum())
        above_surface = int(component[:coarse_surface_y].sum())
        if below_surface < config.min_droplet_area_px2 or above_surface < config.min_droplet_area_px2:
            continue
        if y_max < coarse_surface_y + config.mirror_min_depth_px:
            continue

        score = area + 20 * below_surface + 100 * vertical_span + (x_max - x_min + 1)
        if score > best_score:
            best_label = label
            best_score = score

    if best_label == 0:
        return None

    component = labels == best_label
    bbox = component_bbox(component)
    if bbox is None:
        return None
    return component, bbox


def estimate_vertical_symmetry_y(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, float]:
    """Estimate a horizontal symmetry axis from component row widths.

    The surface calibration frame is expected to contain a droplet and its
    reflection.  Comparing the upper and lower silhouette avoids relying on a
    coarse background edge, which can identify a second substrate boundary
    below the actual surface.
    """
    _x_min, y_min, _x_max, y_max = bbox
    vertical_span = y_max - y_min + 1
    if vertical_span < 5:
        raise ValueError(
            "Droplet/reflection component is too short for symmetry calibration"
        )

    widths = np.array(
        [row_run_width(component, row_y) for row_y in range(y_min, y_max + 1)],
        dtype=np.float32,
    )
    max_width = float(widths.max())
    if max_width <= 0:
        raise ValueError("No foreground rows found in symmetry calibration component")

    quarter_span = max(1, vertical_span // 4)
    center_start = y_min + quarter_span
    center_stop = y_max - quarter_span
    best_y = center_start
    best_error = math.inf

    for center_y in range(center_start, center_stop + 1):
        center_index = center_y - y_min
        half_span = min(center_y - y_min, y_max - center_y)
        if half_span < 1:
            continue

        upper = widths[center_index - half_span : center_index]
        lower = widths[center_index + 1 : center_index + half_span + 1][::-1]
        silhouette_error = float(np.mean(np.abs(upper - lower))) / max_width
        extent_error = abs((center_y - y_min) - (y_max - center_y)) / vertical_span
        error = silhouette_error + extent_error
        if error < best_error:
            best_y = center_y
            best_error = error

    return best_y, best_error


def select_symmetric_calibration_component(
    mask: np.ndarray,
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Find a nearby component without requiring it to cross the coarse edge."""
    labels, count = ndi.label(mask)
    best: (
        tuple[tuple[float, float, int], np.ndarray, tuple[int, int, int, int]]
        | None
    ) = None
    max_edge_gap = max(
        DEFAULT_CALIBRATION_EDGE_GAP_PX,
        4 * max(1, config.morphology_radius_px),
    )
    min_vertical_span = 2 * max(2, config.morphology_radius_px) + 1

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_droplet_area_px2:
            continue

        bbox = component_bbox(component)
        if bbox is None:
            continue
        _x_min, y_min, _x_max, y_max = bbox
        if y_max - y_min + 1 < min_vertical_span:
            continue

        if coarse_surface_y < y_min:
            edge_gap = y_min - coarse_surface_y
        elif coarse_surface_y > y_max:
            edge_gap = coarse_surface_y - y_max
        else:
            edge_gap = 0
        if edge_gap > max_edge_gap:
            continue

        symmetry_y, symmetry_error = estimate_vertical_symmetry_y(component, bbox)
        min_side_area = min(config.min_droplet_area_px2, max(1, area // 10))
        above_area = int(component[:symmetry_y].sum())
        below_area = int(component[symmetry_y + 1 :].sum())
        if above_area < min_side_area or below_area < min_side_area:
            continue

        score = (float(edge_gap), symmetry_error, -area)
        if best is None or score < best[0]:
            best = (score, component, bbox)

    if best is None:
        return None
    return best[1], best[2]


def detect_reflection_mode(
    mask: np.ndarray,
    coarse_surface_y: int,
    config: AnalysisConfig,
) -> str:
    """Conservatively classify a calibration silhouette as mirror or none."""
    if select_calibration_component(mask, coarse_surface_y, config) is not None:
        return "mirror"

    selected = select_symmetric_calibration_component(mask, coarse_surface_y, config)
    if selected is None:
        return "none"
    component, bbox = selected
    _symmetry_y, symmetry_error = estimate_vertical_symmetry_y(component, bbox)
    if symmetry_error <= MAX_REFLECTION_SYMMETRY_ERROR:
        return "mirror"
    return "none"


def row_run_width(component: np.ndarray, row_y: int) -> int:
    bounds = longest_true_run(component[row_y])
    if bounds is None:
        return 0
    x_min, x_max = bounds
    return int(x_max - x_min + 1)


def _contact_vertex_index(profile: np.ndarray, points_outward: bool) -> int:
    """Find a contact tip, or the notch bracketed by two reflection lobes."""
    smoothed = ndi.gaussian_filter1d(profile.astype(np.float32), sigma=2.0)
    outward_signal = -smoothed if points_outward else smoothed
    inward_signal = -outward_signal
    prominence = max(2.0, 0.005 * float(np.ptp(profile)))
    distance = max(3, int(round(0.03 * len(profile))))
    outward, outward_properties = find_peaks(
        outward_signal,
        prominence=prominence,
        distance=distance,
    )
    inward, inward_properties = find_peaks(
        inward_signal,
        prominence=prominence,
        distance=distance,
    )

    bracketed_inward = [
        (index, inward_properties["prominences"][position])
        for position, index in enumerate(inward)
        if np.any(outward < index) and np.any(outward > index)
    ]
    if bracketed_inward:
        return int(max(bracketed_inward, key=lambda item: item[1])[0])
    if outward.size:
        strongest = int(np.argmax(outward_properties["prominences"]))
        return int(outward[strongest])
    return int(np.argmax(outward_signal))


def estimate_contact_line(component: np.ndarray) -> SurfaceLine:
    """Fit the surface through the two contact tips or concave vertices."""
    bbox = component_bbox(component)
    if bbox is None:
        raise ValueError("Cannot estimate a contact line from an empty component")
    x_min, y_min, x_max, y_max = bbox
    row_ys = np.arange(y_min, y_max + 1)
    left_profile = np.empty(len(row_ys), dtype=np.float32)
    right_profile = np.empty(len(row_ys), dtype=np.float32)
    for index, row_y in enumerate(row_ys):
        bounds = longest_true_run(component[row_y], close_size=1)
        if bounds is None:
            raise ValueError("Calibration component contains an empty interior row")
        left_profile[index], right_profile[index] = bounds

    left_index = _contact_vertex_index(left_profile, points_outward=True)
    right_index = _contact_vertex_index(right_profile, points_outward=False)
    left_x = float(left_profile[left_index])
    left_y = float(row_ys[left_index])
    right_x = float(right_profile[right_index])
    right_y = float(row_ys[right_index])
    if right_x <= left_x:
        raise ValueError("Detected contact vertices do not form a valid surface line")

    slope = (right_y - left_y) / (right_x - left_x)
    image_center_x = (component.shape[1] - 1) / 2.0
    center_y = left_y + slope * (image_center_x - left_x)
    return SurfaceLine(center_y, math.degrees(math.atan(slope)))


def calibration_frame_mask(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    structure: np.ndarray,
    weak_threshold: float | None = None,
    weak_reach_px: int = DEFAULT_WEAK_REACH_PX,
) -> np.ndarray:
    matching_file = next(
        (path for path in files if frame_number_from_filename(path) == frame_number),
        None,
    )
    if matching_file is None:
        raise ValueError(
            f"--calibration-frame {frame_number} does not exist in the input sequence"
        )
    return full_foreground_mask(
        read_image(matching_file),
        background,
        threshold,
        structure,
        weak_threshold,
        weak_reach_px,
    )


def detect_reflection_mode_from_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
    weak_threshold: float | None = None,
) -> str:
    mask = calibration_frame_mask(
        files,
        background,
        frame_number,
        threshold,
        structure,
        weak_threshold,
        config.weak_reach_px,
    )
    return detect_reflection_mode(mask, coarse_surface_y, config)


def select_nonreflection_calibration_component(
    mask: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray | None:
    """Select the impacted droplet without using a substrate-edge position."""
    labels, count = ndi.label(mask)
    best_component: np.ndarray | None = None
    best_area = 0
    height, width = mask.shape

    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < config.min_droplet_area_px2 or area <= best_area:
            continue
        bbox = component_bbox(component)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        if x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1:
            continue
        best_component = component
        best_area = area

    return best_component


def calibration_target_center_x(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
    reflection_mode: str,
    weak_threshold: float | None = None,
) -> float | None:
    """Return the calibrated droplet's horizontal centre, when available.

    The calibration frame identifies which droplet in a multi-drop recording
    is the subject of the measurement. Subsequent frames can otherwise select
    an unrelated drop merely because it happens to be closer to the surface.
    """
    mask = calibration_frame_mask(
        files,
        background,
        frame_number,
        threshold,
        structure,
        weak_threshold,
        config.weak_reach_px,
    )
    if reflection_mode == "none":
        component = select_nonreflection_calibration_component(mask, config)
        bbox = component_bbox(component) if component is not None else None
    else:
        selected = select_calibration_component(mask, coarse_surface_y, config)
        if selected is None:
            selected = select_symmetric_calibration_component(
                mask, coarse_surface_y, config
            )
        bbox = selected[1] if selected is not None else None
    if bbox is None:
        return None
    return (bbox[0] + bbox[2]) / 2.0


def estimate_lower_contour_line(
    component: np.ndarray,
    angle_override_deg: float | None,
) -> SurfaceLine:
    """Fit a tangent-like surface line to the lowest stable contour segment."""
    bbox = component_bbox(component)
    if bbox is None:
        raise ValueError("Cannot estimate a lower contour from an empty component")
    x_min, _y_min, x_max, _y_max = bbox
    contour_x = []
    contour_y = []
    for x in range(x_min, x_max + 1):
        ys = np.flatnonzero(component[:, x])
        if ys.size:
            contour_x.append(x)
            contour_y.append(int(ys[-1]))

    xs = np.asarray(contour_x, dtype=np.float64)
    ys = np.asarray(contour_y, dtype=np.float64)
    if len(xs) < 2:
        raise ValueError("Droplet lower contour contains fewer than two points")

    # Image y increases downward. Restrict the fit to the lowest 20% of the
    # per-column contour so curved side walls do not pull the line upward.
    cutoff = float(np.percentile(ys, 80))
    lower = ys >= cutoff
    if int(lower.sum()) < 5:
        lowest_indices = np.argsort(ys)[-min(5, len(ys)) :]
        fit_x = xs[lowest_indices]
        fit_y = ys[lowest_indices]
    else:
        fit_x = xs[lower]
        fit_y = ys[lower]

    if len(fit_x) >= 3:
        for _ in range(4):
            slope, intercept = np.polyfit(fit_x, fit_y, 1)
            residual = fit_y - (slope * fit_x + intercept)
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            inlier_limit = max(1.5, 3.0 * 1.4826 * mad)
            inliers = np.abs(residual - median) <= inlier_limit
            if int(inliers.sum()) < 10 or bool(np.all(inliers)):
                break
            fit_x = fit_x[inliers]
            fit_y = fit_y[inliers]
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        center_y = float(slope * ((component.shape[1] - 1) / 2.0) + intercept)
        detected_angle = math.degrees(math.atan(float(slope)))
        if abs(detected_angle) > 5.0:
            center_y = float(np.median(fit_y))
            detected_angle = 0.0
    else:
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        center_y = float(slope * ((component.shape[1] - 1) / 2.0) + intercept)
        detected_angle = math.degrees(math.atan(float(slope)))

    angle = detected_angle if angle_override_deg is None else float(angle_override_deg)
    return SurfaceLine(center_y, angle)


def estimate_surface_line_from_nonreflection_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    config: AnalysisConfig,
    structure: np.ndarray,
    angle_override_deg: float | None = None,
    weak_threshold: float | None = None,
) -> SurfaceLine:
    """Fit the surface to the impacted droplet's lower contour."""
    mask = calibration_frame_mask(
        files,
        background,
        frame_number,
        threshold,
        structure,
        weak_threshold,
        config.weak_reach_px,
    )
    component = select_nonreflection_calibration_component(mask, config)
    if component is None:
        raise ValueError(
            "Could not find a non-border droplet component in "
            f"surface calibration frame {frame_number}"
        )
    return estimate_lower_contour_line(component, angle_override_deg)


def estimate_surface_line_from_symmetry_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
    angle_override_deg: float | None = None,
    weak_threshold: float | None = None,
) -> SurfaceLine:
    mask = calibration_frame_mask(
        files,
        background,
        frame_number,
        threshold,
        structure,
        weak_threshold,
        config.weak_reach_px,
    )
    selected = select_calibration_component(mask, coarse_surface_y, config)
    if selected is None:
        selected = select_symmetric_calibration_component(mask, coarse_surface_y, config)
        if selected is None:
            raise ValueError(
                "Could not find a droplet/reflection component in surface "
                f"calibration frame {frame_number}"
            )
        component, _bbox = selected
    else:
        component, _bbox = selected

    detected_line = estimate_contact_line(component)
    if angle_override_deg is None:
        return detected_line
    return SurfaceLine(detected_line.center_y, float(angle_override_deg))


def estimate_surface_y_from_symmetry_frame(
    files: list[Path],
    background: np.ndarray,
    frame_number: int,
    threshold: float,
    coarse_surface_y: int,
    config: AnalysisConfig,
    structure: np.ndarray,
) -> int:
    line = estimate_surface_line_from_symmetry_frame(
        files,
        background,
        frame_number,
        threshold,
        coarse_surface_y,
        config,
        structure,
        angle_override_deg=0.0,
    )
    return line.center_y_int()
