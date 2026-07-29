from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .imaging import (
    build_background,
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
)
from .models import (
    DEFAULT_THRESHOLD_SAMPLE_FRAMES,
    DEFAULT_SURFACE_ANGLE_DEG,
    AnalysisConfig,
    FrameMeasurement,
    SurfaceLine,
)
from .visualization import clear_debug_overlays, write_debug_overlay


def resolve_weak_threshold(config: AnalysisConfig, threshold: float) -> float:
    if config.weak_threshold is not None:
        return float(config.weak_threshold)
    return threshold * config.weak_threshold_fraction


def analyze_sequence(config: AnalysisConfig) -> list[FrameMeasurement]:
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
    first_frame_number = frame_number_from_filename(files[0])
    background = build_background(files)
    coarse_surface_y = estimate_surface_y(
        background,
        config.surface_search_start_px,
        config.surface_drop_delta,
    )
    structure = make_structure(config.morphology_radius_px)
    preliminary_threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(
            files,
            background,
            coarse_surface_y,
            DEFAULT_THRESHOLD_SAMPLE_FRAMES,
            config.min_foreground_delta,
        )
    )
    preliminary_weak_threshold = resolve_weak_threshold(config, preliminary_threshold)
    calibrated_surface_line: SurfaceLine | None = None
    resolved_reflection_mode = config.reflection_mode
    if config.surface_frame is not None:
        if resolved_reflection_mode == "auto":
            resolved_reflection_mode = detect_reflection_mode_from_frame(
                files,
                background,
                config.surface_frame,
                preliminary_threshold,
                coarse_surface_y,
                config,
                structure,
                weak_threshold=preliminary_weak_threshold,
            )
        if resolved_reflection_mode == "mirror":
            calibrated_surface_line = estimate_surface_line_from_symmetry_frame(
                files,
                background,
                config.surface_frame,
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
                config.surface_frame,
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

    if calibrated_surface_line is not None:
        surface_angle_deg = calibrated_surface_line.angle_deg
    elif config.surface_angle_deg is not None:
        surface_angle_deg = config.surface_angle_deg
    else:
        surface_angle_deg = DEFAULT_SURFACE_ANGLE_DEG

    if config.surface_y is not None:
        surface_line = SurfaceLine(float(int(config.surface_y)), surface_angle_deg)
    elif calibrated_surface_line is not None:
        # The line calibrated from --calibration-frame is the single fixed
        # reference used verbatim for every frame in the sequence.
        surface_line = calibrated_surface_line
    else:
        surface_line = SurfaceLine(float(coarse_surface_y), surface_angle_deg)

    threshold = (
        float(config.threshold)
        if config.threshold is not None
        else estimate_threshold(
            files,
            background,
            surface_line.center_y_int(),
            DEFAULT_THRESHOLD_SAMPLE_FRAMES,
            config.min_foreground_delta,
        )
    )
    weak_threshold = resolve_weak_threshold(config, threshold)

    if config.debug_dir is not None:
        clear_debug_overlays(config.debug_dir)

    pending = []
    impact_frame: int | None = None
    reference_frame: int | None = None
    reference_measurement = None

    for file_path in files:
        frame_number = frame_number_from_filename(file_path)
        image = read_image(file_path)
        mask = foreground_mask(
            image,
            background,
            surface_line,
            threshold,
            effective_config,
            structure,
            weak_threshold=weak_threshold,
        )
        measurement = component_measurement(mask, surface_line, effective_config)
        is_touching = touches_surface(measurement, effective_config)
        if impact_frame is None and is_touching:
            impact_frame = frame_number
        is_reference_frame = False
        if (
            reference_measurement is None
            and measurement.equivalent_diameter_px is not None
            and measurement.surface_clearance_px is not None
            and measurement.surface_clearance_px
            <= measurement.equivalent_diameter_px / 2.0
        ):
            reference_frame = frame_number
            reference_measurement = measurement
            is_reference_frame = True
        pending.append((frame_number, file_path, measurement))

        if config.debug_dir is not None and (
            frame_number == first_frame_number
            or frame_number % config.debug_every == 0
            or is_reference_frame
            or (is_touching and abs(frame_number - (impact_frame or frame_number)) <= 2)
        ):
            debug_path = config.debug_dir / f"{frame_number:06d}.png"
            write_debug_overlay(
                debug_path,
                image,
                measurement.mask if reference_measurement is not None else None,
                surface_line,
                (
                    measurement.contact_bbox
                    if effective_config.hydrophobic
                    else measurement.bbox
                ) if reference_measurement is not None else None,
                frame_number,
                spreading_diameter_mm=(
                    measurement.spreading_diameter_px * config.pixel_size_mm
                ),
                contact_line_diameter_mm=(
                    measurement.contact_line_diameter_px * config.pixel_size_mm
                    if measurement.contact_line_diameter_px is not None
                    else None
                ),
                rim_to_rim_diameter_mm=(
                    measurement.rim_to_rim_diameter_px * config.pixel_size_mm
                    if measurement.rim_to_rim_diameter_px is not None
                    else None
                ),
                surface_parallel_line=(
                    measurement.surface_parallel_line
                    if reference_measurement is not None
                    else None
                ),
                surface_normal_line=(
                    measurement.surface_normal_line
                    if reference_measurement is not None
                    else None
                ),
                initial_equivalent_diameter_mm=(
                    reference_measurement.equivalent_diameter_px * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.equivalent_diameter_px is not None
                    else None
                ),
                initial_surface_parallel_diameter_mm=(
                    reference_measurement.surface_parallel_diameter_px
                    * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.surface_parallel_diameter_px is not None
                    else None
                ),
                initial_surface_normal_diameter_mm=(
                    reference_measurement.surface_normal_diameter_px
                    * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.surface_normal_diameter_px is not None
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

        spreading_diameter_mm = measurement.spreading_diameter_px * config.pixel_size_mm
        measurements.append(
            FrameMeasurement(
                frame_number=frame_number,
                filename=file_path.name,
                time_ms=frame_offset / config.fps * 1000.0,
                spreading_diameter_px=measurement.spreading_diameter_px,
                spreading_diameter_mm=spreading_diameter_mm,
                component_area_px=measurement.area_px,
                surface_y=surface_line.center_y_int(),
                impact_frame=impact_frame,
                fps=config.fps,
                pixel_size_mm=config.pixel_size_mm,
                surface_frame=config.surface_frame,
                contact_line_diameter_px=measurement.contact_line_diameter_px,
                contact_line_diameter_mm=(
                    measurement.contact_line_diameter_px * config.pixel_size_mm
                    if measurement.contact_line_diameter_px is not None
                    else None
                ),
                rim_to_rim_diameter_px=measurement.rim_to_rim_diameter_px,
                rim_to_rim_diameter_mm=(
                    measurement.rim_to_rim_diameter_px * config.pixel_size_mm
                    if measurement.rim_to_rim_diameter_px is not None
                    else None
                ),
                reference_frame=reference_frame,
                initial_equivalent_diameter_px=(
                    reference_measurement.equivalent_diameter_px
                    if reference_measurement is not None
                    else None
                ),
                initial_equivalent_diameter_mm=(
                    reference_measurement.equivalent_diameter_px * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.equivalent_diameter_px is not None
                    else None
                ),
                initial_surface_parallel_diameter_px=(
                    reference_measurement.surface_parallel_diameter_px
                    if reference_measurement is not None
                    else None
                ),
                initial_surface_parallel_diameter_mm=(
                    reference_measurement.surface_parallel_diameter_px
                    * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.surface_parallel_diameter_px is not None
                    else None
                ),
                initial_surface_normal_diameter_px=(
                    reference_measurement.surface_normal_diameter_px
                    if reference_measurement is not None
                    else None
                ),
                initial_surface_normal_diameter_mm=(
                    reference_measurement.surface_normal_diameter_px
                    * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.surface_normal_diameter_px is not None
                    else None
                ),
                surface_clearance_px=(
                    reference_measurement.surface_clearance_px
                    if reference_measurement is not None
                    else None
                ),
                surface_clearance_mm=(
                    reference_measurement.surface_clearance_px * config.pixel_size_mm
                    if reference_measurement is not None
                    and reference_measurement.surface_clearance_px is not None
                    else None
                ),
            )
        )

    return measurements
