from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
import warnings

import numpy as np

from .imaging import (
    build_background,
    calibration_target_center_x,
    component_measurement,
    detect_reflection_mode_from_frame,
    estimate_surface_y,
    estimate_surface_line_from_nonreflection_frame,
    estimate_surface_line_from_symmetry_frame,
    estimate_threshold,
    find_image_files,
    frame_number_from_filename,
    foreground_mask,
    make_structure,
    read_image,
    touches_surface,
    threshold_sampling_anchor,
)
from .models import (
    DEFAULT_THRESHOLD_SAMPLE_FRAMES,
    DEFAULT_THRESHOLD_PRE_IMPACT_FRAMES,
    DEFAULT_THRESHOLD_POST_IMPACT_FRAMES,
    DEFAULT_SURFACE_ANGLE_DEG,
    AnalysisConfig,
    FrameMeasurement,
    ComponentMeasurement,
    SurfaceLine,
)
from .visualization import clear_debug_overlays, write_debug_overlay


def resolve_weak_threshold(config: AnalysisConfig, threshold: float) -> float:
    if config.weak_threshold is not None:
        return float(config.weak_threshold)
    return threshold * config.weak_threshold_ratio


def select_sequence_files(config: AnalysisConfig) -> list[Path]:
    all_files = find_image_files(config.input_dir)
    if config.start_frame is not None and config.start_frame < 1:
        raise ValueError("start_frame must be at least 1")
    if config.end_frame is not None and config.end_frame < 1:
        raise ValueError("end_frame must be at least 1")
    if (
        config.start_frame is not None
        and config.end_frame is not None
        and config.start_frame > config.end_frame
    ):
        raise ValueError("start_frame must be less than or equal to end_frame")

    files = all_files
    if config.start_frame is not None or config.end_frame is not None:
        files_in_range = []
        for path in files:
            frame_number = frame_number_from_filename(path)
            after_start = config.start_frame is None or frame_number >= config.start_frame
            before_end = config.end_frame is None or frame_number <= config.end_frame
            if after_start and before_end:
                files_in_range.append(path)
        files = files_in_range
        if not files:
            bounds = []
            if config.start_frame is not None:
                bounds.append(f"at or after frame {config.start_frame}")
            if config.end_frame is not None:
                bounds.append(f"at or before frame {config.end_frame}")
            raise ValueError(f"No input frames found {' and '.join(bounds)}")
    return files


@dataclass(frozen=True)
class PreparedSequence:
    config: AnalysisConfig
    files: list[Path]
    background: np.ndarray
    surface_line: SurfaceLine
    threshold: float
    weak_threshold: float
    structure: np.ndarray
    target_center_x: float | None
    threshold_anchor_frame: int | None = None
    threshold_sampling_method: str = "manual"
    threshold_sample_frames: tuple[int, ...] = ()

    @property
    def surface_is_calibrated(self) -> bool:
        return (self.config.surface_y_px is not None
                or self.config.surface_calibration_frame is not None)

    def measure(self, file_path: Path) -> tuple[np.ndarray, ComponentMeasurement]:
        image = read_image(file_path)
        mask = foreground_mask(
            image, self.background, self.surface_line, self.threshold,
            self.config, self.structure, weak_threshold=self.weak_threshold,
        )
        return image, component_measurement(
            mask, self.surface_line, self.config, self.target_center_x
        )


def is_reference_measurement(measurement: ComponentMeasurement) -> bool:
    return (
        measurement.d_eq_volume_px is not None
        and measurement.surface_clearance_px is not None
        and measurement.surface_clearance_px <= measurement.d_eq_volume_px / 2.0
    )


def prepare_sequence(
    config: AnalysisConfig,
    files: list[Path] | None = None,
    background: np.ndarray | None = None,
) -> PreparedSequence:
    files = select_sequence_files(config) if files is None else files
    reference_frame = config.droplet_size_ref_frame
    if reference_frame is not None and reference_frame not in {
        frame_number_from_filename(path) for path in files
    }:
        raise ValueError("Droplet size reference frame must be an existing image number in the working range.")
    background = build_background(files) if background is None else background
    coarse_surface_y = estimate_surface_y(
        background,
        config.surface_search_start_px,
        config.surface_drop_delta,
    )
    structure = make_structure(config.morphology_radius_px)
    preliminary_threshold = (
        float(config.strong_threshold)
        if config.strong_threshold is not None
        else estimate_threshold(
            files,
            background,
            coarse_surface_y,
            DEFAULT_THRESHOLD_SAMPLE_FRAMES,
            config.min_strong_threshold,
        )
    )
    preliminary_weak_threshold = resolve_weak_threshold(config, preliminary_threshold)
    calibrated_surface_line: SurfaceLine | None = None
    resolved_reflection_mode = config.reflection_mode
    if config.surface_calibration_frame is not None:
        if resolved_reflection_mode == "auto":
            resolved_reflection_mode = detect_reflection_mode_from_frame(
                files,
                background,
                config.surface_calibration_frame,
                preliminary_threshold,
                coarse_surface_y,
                config,
                structure,
                weak_threshold=preliminary_weak_threshold,
            )
        if config.surface_y_px is not None and config.surface_angle_deg is not None:
            # A complete manual line must not depend on successful silhouette
            # calibration. The selected frame can still identify the target.
            calibrated_surface_line = SurfaceLine(config.surface_y_px, config.surface_angle_deg)
        elif resolved_reflection_mode == "mirror":
            calibrated_surface_line = estimate_surface_line_from_symmetry_frame(
                files,
                background,
                config.surface_calibration_frame,
                preliminary_threshold,
                coarse_surface_y,
                config,
                structure,
                angle_override_deg=config.surface_angle_deg,
                weak_threshold=preliminary_weak_threshold,
            )
        else:
            calibrated_surface_line = estimate_surface_line_from_nonreflection_frame(
                files,
                background,
                config.surface_calibration_frame,
                preliminary_threshold,
                config,
                structure,
                angle_override_deg=config.surface_angle_deg,
                weak_threshold=preliminary_weak_threshold,
            )
    elif resolved_reflection_mode == "auto":
        # Without a calibration frame there is no silhouette evidence to
        # classify, so mirror mode is assumed.
        resolved_reflection_mode = "mirror"

    effective_config = replace(config, reflection_mode=resolved_reflection_mode)
    target_center_x = (
        calibration_target_center_x(
            files,
            background,
            config.surface_calibration_frame,
            preliminary_threshold,
            coarse_surface_y,
            config,
            structure,
            resolved_reflection_mode,
            weak_threshold=preliminary_weak_threshold,
        )
        if config.surface_calibration_frame is not None
        else None
    )

    if calibrated_surface_line is not None:
        surface_angle_deg = calibrated_surface_line.angle_deg
    elif config.surface_angle_deg is not None:
        surface_angle_deg = config.surface_angle_deg
    else:
        surface_angle_deg = DEFAULT_SURFACE_ANGLE_DEG

    if config.surface_y_px is not None:
        surface_line = SurfaceLine(float(int(config.surface_y_px)), surface_angle_deg)
    elif calibrated_surface_line is not None:
        # The line calibrated from --surface-calibration-frame is the single fixed
        # reference used verbatim for every frame in the sequence.
        surface_line = calibrated_surface_line
    else:
        surface_line = SurfaceLine(float(coarse_surface_y), surface_angle_deg)

    anchor_index = None
    sampling_method = "manual"
    sample_files: list[Path] = []
    threshold = preliminary_threshold
    if config.strong_threshold is None:
        anchor_index, sampling_method = threshold_sampling_anchor(
            files, background, surface_line, preliminary_threshold,
            preliminary_weak_threshold, effective_config, structure,
            config.surface_y_px is not None or calibrated_surface_line is not None,
            target_center_x,
        )
        if anchor_index is None:
            indices = np.unique(np.linspace(
                0, len(files) - 1, min(DEFAULT_THRESHOLD_SAMPLE_FRAMES, len(files)), dtype=int
            ))
            sample_files = [files[index] for index in indices]
        else:
            sample_files = files[
                max(0, anchor_index - DEFAULT_THRESHOLD_PRE_IMPACT_FRAMES):
                anchor_index + DEFAULT_THRESHOLD_POST_IMPACT_FRAMES + 1
            ]
        threshold = estimate_threshold(
            sample_files, background, surface_line.center_y_int(), len(sample_files),
            config.min_strong_threshold, surface_line=surface_line,
        )
    weak_threshold = resolve_weak_threshold(config, threshold)
    return PreparedSequence(
        effective_config, files, background, surface_line, threshold,
        weak_threshold, structure, target_center_x,
        frame_number_from_filename(files[anchor_index]) if anchor_index is not None else None,
        sampling_method, tuple(frame_number_from_filename(path) for path in sample_files),
    )


def analyze_sequence(
    config: AnalysisConfig,
    progress: Callable[[int, int], None] | None = None,
) -> list[FrameMeasurement]:
    prepared = prepare_sequence(config)
    if not prepared.surface_is_calibrated:
        warnings.warn(
            "Surface line is an uncalibrated background estimate. Select two surface "
            "points in the GUI or supply --surface-y-px and --surface-angle-deg "
            "before relying on contact width and impact time.",
            UserWarning, stacklevel=2,
        )
    files = prepared.files
    first_frame_number = frame_number_from_filename(files[0])
    surface_line = prepared.surface_line
    effective_config = prepared.config

    if config.debug_dir is not None:
        clear_debug_overlays(config.debug_dir)

    pending = []
    impact_frame: int | None = None
    droplet_size_ref_frame: int | None = None
    ref_measurement = None

    for index, file_path in enumerate(files):
        if progress is not None:
            progress(index, len(files))
        frame_number = frame_number_from_filename(file_path)
        image, measurement = prepared.measure(file_path)
        is_touching = touches_surface(measurement, effective_config)
        if impact_frame is None and is_touching:
            impact_frame = frame_number
        is_ref_frame = False
        selected_reference = (
            frame_number == config.droplet_size_ref_frame
            if config.droplet_size_ref_frame is not None
            else is_reference_measurement(measurement)
        )
        if ref_measurement is None and selected_reference:
            if measurement.d_eq_volume_px is None or measurement.d_eq_area_px is None:
                raise ValueError("Droplet size reference frame must contain a measurable droplet.")
            droplet_size_ref_frame = frame_number
            ref_measurement = measurement
            is_ref_frame = True
        pending.append((frame_number, file_path, measurement))

        if config.debug_dir is not None and (
            frame_number == first_frame_number
            or frame_number % config.debug_interval == 0
            or is_ref_frame
            or (is_touching and abs(frame_number - (impact_frame or frame_number)) <= 2)
        ):
            debug_path = config.debug_dir / f"{frame_number:06d}.png"
            write_debug_overlay(
                debug_path,
                image,
                measurement.mask,
                surface_line,
                measurement.contact_bbox,
                frame_number,
                d_spread_mm=(
                    measurement.d_spread_px * config.pixel_size_mm_per_px
                ),
                rim_bbox=measurement.rim_bbox,
                d_contact_mm=(
                    measurement.d_contact_px * config.pixel_size_mm_per_px
                    if measurement.d_contact_px is not None
                    else None
                ),
                d_rim_mm=(
                    measurement.d_rim_px * config.pixel_size_mm_per_px
                    if measurement.d_rim_px is not None
                    else None
                ),
            )

    measurements: list[FrameMeasurement] = []
    for frame_number, file_path, measurement in pending:
        if impact_frame is None:
            frame_offset = frame_number - first_frame_number
        elif config.time_zero == "impact":
            frame_offset = frame_number - impact_frame
        else:
            frame_offset = frame_number - first_frame_number

        if not config.include_pre_impact and impact_frame is not None and frame_number < impact_frame:
            continue

        d_spread_mm = measurement.d_spread_px * config.pixel_size_mm_per_px
        measurements.append(
            FrameMeasurement(
                frame_number=frame_number,
                filename=file_path.name,
                time_ms=frame_offset / config.fps * 1000.0,
                d_spread_px=measurement.d_spread_px,
                d_spread_mm=d_spread_mm,
                droplet_area_px2=measurement.area_px2,
                surface_y_px=surface_line.center_y_int(),
                impact_frame=impact_frame,
                fps=config.fps,
                pixel_size_mm_per_px=config.pixel_size_mm_per_px,
                surface_calibration_frame=config.surface_calibration_frame,
                d_contact_px=measurement.d_contact_px,
                d_contact_mm=(
                    measurement.d_contact_px * config.pixel_size_mm_per_px
                    if measurement.d_contact_px is not None
                    else None
                ),
                d_rim_px=measurement.d_rim_px,
                d_rim_mm=(
                    measurement.d_rim_px * config.pixel_size_mm_per_px
                    if measurement.d_rim_px is not None
                    else None
                ),
                droplet_size_ref_frame=droplet_size_ref_frame,
                ref_d_eq_volume_px=(
                    ref_measurement.d_eq_volume_px
                    if ref_measurement is not None
                    else None
                ),
                ref_d_eq_volume_mm=(
                    ref_measurement.d_eq_volume_px * config.pixel_size_mm_per_px
                    if ref_measurement is not None
                    and ref_measurement.d_eq_volume_px is not None
                    else None
                ),
                ref_d_eq_area_px=(
                    ref_measurement.d_eq_area_px
                    if ref_measurement is not None
                    else None
                ),
                ref_d_eq_area_mm=(
                    ref_measurement.d_eq_area_px * config.pixel_size_mm_per_px
                    if ref_measurement is not None
                    and ref_measurement.d_eq_area_px is not None
                    else None
                ),
                ref_d_parallel_px=(
                    ref_measurement.d_parallel_px
                    if ref_measurement is not None
                    else None
                ),
                ref_d_parallel_mm=(
                    ref_measurement.d_parallel_px
                    * config.pixel_size_mm_per_px
                    if ref_measurement is not None
                    and ref_measurement.d_parallel_px is not None
                    else None
                ),
                ref_d_normal_px=(
                    ref_measurement.d_normal_px
                    if ref_measurement is not None
                    else None
                ),
                ref_d_normal_mm=(
                    ref_measurement.d_normal_px
                    * config.pixel_size_mm_per_px
                    if ref_measurement is not None
                    and ref_measurement.d_normal_px is not None
                    else None
                ),
                ref_clearance_px=(
                    ref_measurement.surface_clearance_px
                    if ref_measurement is not None
                    else None
                ),
                ref_clearance_mm=(
                    ref_measurement.surface_clearance_px * config.pixel_size_mm_per_px
                    if ref_measurement is not None
                    and ref_measurement.surface_clearance_px is not None
                    else None
                ),
            )
        )

    if progress is not None:
        progress(len(files), len(files))
    return measurements
