from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

from .analysis import analyze_sequence
from .models import (
    DEFAULT_FPS,
    DEFAULT_MIN_STRONG_CHANGE_THRESHOLD,
    DEFAULT_MIRROR_MIN_DEPTH_PX,
    DEFAULT_PIXEL_SIZE_MM_PER_PX,
    DEFAULT_CONTACT_BAND_HALF_HEIGHT_PX,
    DEFAULT_TOP_EXCLUSION_RATIO,
    DEFAULT_WEAK_GROWTH_STEPS,
    DEFAULT_WEAK_CHANGE_THRESHOLD_RATIO,
    SURFACE_CALIBRATION_METHODS,
    OUTPUT_CSV_FILENAME,
    AnalysisConfig,
)
from .output import RUN_PARAMS_FILENAME, write_csv, write_run_params


CONFIG_FILENAME = "cv_config.txt"
PROJECT_DIR = Path(__file__).resolve().parent.parent


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def zero_or_one(value: str) -> int:
    parsed = int(value)
    if parsed not in (0, 1):
        raise argparse.ArgumentTypeError("value must be 0 or 1")
    return parsed


def unit_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in the range (0, 1]")
    return parsed


def default_output_dir(input_dir: Path) -> Path:
    return PROJECT_DIR / "outputs" / input_dir.resolve().name


def exclusion_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in the range [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Measure droplet spreading diameter from high-speed TIFF sequences. "
            "Options in <input_dir>/cv_config.txt are loaded automatically; "
            "explicit command-line options take precedence."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("sourcedata/example"),
        help="Directory containing TIFF frames. Default: sourcedata/example",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for cv_output.csv, run_params.json, and debug_overlays/. "
             "Default: <project-dir>/outputs/<input-folder>",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        type=Path,
        metavar="PATH",
        default=None,
        help="CSV path override; takes precedence over --output-dir. "
             "Default: <output-dir>/cv_output.csv",
    )
    parser.add_argument(
        "--substrate",
        default=None,
        help="Surface name recorded as experiment metadata; does not affect measurements.",
    )
    parser.add_argument(
        "--hydrophilicity",
        choices=("SHI", "HI", "HO", "SHO"),
        default=None,
        help="Surface wettability recorded as experiment metadata; does not affect measurements.",
    )
    parser.add_argument(
        "--substrate_temp_deg_c",
        type=float,
        default=None,
        help="Surface temperature in degrees Celsius, recorded as experiment metadata only.",
    )
    parser.add_argument(
        "--start-frame",
        type=positive_int,
        default=None,
        help=(
            "Only process frames from this filename frame number (inclusive). "
            "Default: no lower limit."
        ),
    )
    parser.add_argument(
        "--end-frame",
        type=positive_int,
        default=None,
        help=(
            "Only process frames up to this filename frame number (inclusive). "
            "Default: no upper limit."
        ),
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=DEFAULT_FPS,
        help="Camera frame rate in fps. Default: 1000",
    )
    parser.add_argument(
        "--pixel-size-mm-per-px",
        dest="pixel_size_mm_per_px",
        type=positive_float,
        default=DEFAULT_PIXEL_SIZE_MM_PER_PX,
        help="Pixel size in mm/px. Default: 0.01",
    )
    parser.add_argument(
        "--surface-y-px",
        dest="surface_y_px",
        type=nonnegative_int,
        metavar="Y",
        default=None,
        help="Override surface-line y at the image's horizontal midpoint in px.",
    )
    parser.add_argument(
        "--surface-calibration-frame",
        dest="surface_calibration_frame",
        type=nonnegative_int,
        metavar="FRAME",
        default=None,
        help=(
            "Frame number containing an impacted droplet, used to calibrate "
            "the surface line according to --surface-calibration-method."
        ),
    )
    parser.add_argument(
        "--droplet-size-ref-frame",
        type=nonnegative_int,
        metavar="FRAME",
        default=None,
        help=(
            "Frame number used for both droplet size reference diameters. "
            "Default: first frame with substrate clearance <= half the volume-equivalent diameter."
        ),
    )
    parser.add_argument(
        "--surface-calibration-method",
        choices=SURFACE_CALIBRATION_METHODS,
        default="auto",
        help=(
            "Surface calibration method: auto selects a method, mirror uses reflection "
            "symmetry, none uses the droplet lower contour without mirror symmetry. "
            "This does not enable or disable shadow suppression. Default: auto."
        ),
    )
    parser.add_argument(
        "--surface-angle-deg",
        type=float,
        default=None,
        help=(
            "Override surface angle in clockwise deg. By default the angle is "
            "detected from --surface-calibration-frame, or 0 when no calibration frame is given."
        ),
    )
    parser.add_argument(
        "--strong-change-threshold",
        dest="strong_change_threshold",
        type=positive_float,
        metavar="VALUE",
        default=None,
        help="Strong threshold on absolute change from the clean background in gray levels; not an image-brightness cutoff.",
    )
    parser.add_argument(
        "--weak-change-threshold",
        dest="weak_change_threshold",
        type=positive_float,
        metavar="VALUE",
        default=None,
        help=(
            "Weak threshold in gray levels for extending detections across "
            "faint transparent liquid. Pixels above this value count as "
            "foreground only when connected to a pixel above --strong-change-threshold. "
            "Default: --weak-change-threshold-ratio of the effective threshold."
        ),
    )
    parser.add_argument(
        "--weak-change-threshold-ratio",
        dest="weak_change_threshold_ratio",
        type=unit_fraction,
        metavar="RATIO",
        default=DEFAULT_WEAK_CHANGE_THRESHOLD_RATIO,
        help=(
            "Weak threshold as a fraction of the effective strong threshold, "
            "used when --weak-change-threshold is not given. Default: 0.3"
        ),
    )
    parser.add_argument(
        "--mirror-min-depth-px",
        type=nonnegative_int,
        default=DEFAULT_MIRROR_MIN_DEPTH_PX,
        metavar="DISTANCE",
        help="Minimum mirror candidate depth below the coarse surface line in px. Default: 50",
    )
    parser.add_argument(
        "--approach-shadow-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limit near-surface shadows using the approaching drop's contour and two preceding frames. Default: enabled.",
    )
    parser.add_argument(
        "--weak-growth-steps",
        dest="weak_growth_steps",
        type=nonnegative_int,
        metavar="DISTANCE",
        default=DEFAULT_WEAK_GROWTH_STEPS,
        help=(
            "Maximum 4-neighbor growth steps through weak-change pixels from a "
            "strong detection, before outline cleanup; not a Euclidean radius. "
            "0 disables weak extension. Default: 40"
        ),
    )
    parser.add_argument(
        "--min-strong-change-threshold",
        dest="min_strong_change_threshold",
        type=positive_float,
        metavar="VALUE",
        default=DEFAULT_MIN_STRONG_CHANGE_THRESHOLD,
        help="Lower bound for automatic threshold. Default: 200",
    )
    parser.add_argument(
        "--min-component-area-px2",
        dest="min_component_area_px2",
        type=positive_float,
        metavar="AREA",
        default=250,
        help="Minimum liquid component area in px2.",
    )
    parser.add_argument(
        "--morphology-radius-px",
        type=nonnegative_int,
        default=3,
        help="Elliptical cleanup-kernel radius in px (kernel width 2*radius+1). 0 disables gap/speck morphology; enclosed-hole filling remains. Default: 3",
    )
    parser.add_argument(
        "--contact-band-half-height-px",
        type=nonnegative_int,
        default=DEFAULT_CONTACT_BAND_HALF_HEIGHT_PX,
        help=(
            "Vertical half-height in px of the contact-measurement band at each x; "
            "not the full band thickness or surface-normal distance. Default: 1"
        ),
    )
    parser.add_argument(
        "--top-exclusion-ratio",
        type=exclusion_ratio,
        default=DEFAULT_TOP_EXCLUSION_RATIO,
        help=(
            "Fraction of image height excluded from droplet detection at the top. "
            "Range: 0 to 1; 0 disables exclusion, 1 excludes all rows. "
            "Background and surface calibration use the full image. Default: 0.25"
        ),
    )
    parser.add_argument(
        "--foreground-depth-px",
        dest="foreground_depth_px",
        type=nonnegative_int,
        metavar="DEPTH",
        default=220,
        help="Candidate-search depth vertically below the surface line in px; final reported area is clipped to the gas side. Default: 220",
    )
    parser.add_argument(
        "--min-gas-area-px2",
        dest="min_gas_area_px2",
        type=nonnegative_int,
        metavar="AREA",
        default=30,
        help=(
            "Minimum total detected area on or above the surface in px2 for a "
            "selected component; applies to detached drops too. Not contact-line "
            "width. Default: 30"
        ),
    )
    parser.add_argument(
        "--rim-to-rim",
        dest="rim_to_rim",
        type=zero_or_one,
        default=0,
        metavar="{0,1}",
        help=(
            "Use rim-to-rim width (1), instead of "
            "the surface contact-line width (0). Default: 0"
        ),
    )
    parser.add_argument(
        "--min-impact-width-px",
        type=nonnegative_int,
        default=30,
        help=(
            "Minimum contact-line diameter in px for a frame to be marked as "
            "the impact frame. Default: 30"
        ),
    )
    parser.add_argument(
        "--include-pre-impact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include frames before the detected impact frame in the CSV.",
    )
    parser.add_argument(
        "--time-zero",
        choices=("impact", "first-frame"),
        default="impact",
        help="Set time origin. Default: impact",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Overlay folder override; takes precedence over --output-dir. "
             "Default: <output-dir>/debug_overlays",
    )
    parser.add_argument(
        "--debug-interval",
        dest="debug_interval",
        type=positive_int,
        metavar="FRAMES",
        default=1,
        help="Write one debug overlay every N frames.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    command_line_args = parser.parse_args(raw_args)
    config_path = command_line_args.input_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return command_line_args

    try:
        config_tokens = shlex.split(
            config_path.read_text(encoding="utf-8"),
            comments=True,
            posix=True,
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(f"cannot read {config_path}: {error}")

    if "-h" in config_tokens or "--help" in config_tokens:
        parser.error(f"help options are not allowed in {config_path}")

    # Supplying the already resolved input directory first also makes a stray
    # positional value in the config file an argparse error. All remaining
    # values from this parse become defaults for the final CLI parse.
    config_args = parser.parse_args([str(command_line_args.input_dir), *config_tokens])
    config_defaults = vars(config_args).copy()
    config_defaults.pop("input_dir")
    parser.set_defaults(**config_defaults)
    return parser.parse_args(raw_args)


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    output_dir = (args.output_dir or (args.output_csv.parent if args.output_csv else None)
                  or (args.debug_dir.parent if args.debug_dir else None)
                  or default_output_dir(args.input_dir))
    return AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=output_dir,
        output_csv=args.output_csv or output_dir / OUTPUT_CSV_FILENAME,
        fps=args.fps,
        pixel_size_mm_per_px=args.pixel_size_mm_per_px,
        surface_y_px=args.surface_y_px,
        surface_calibration_frame=args.surface_calibration_frame,
        droplet_size_ref_frame=args.droplet_size_ref_frame,
        surface_angle_deg=args.surface_angle_deg,
        surface_calibration_method=args.surface_calibration_method,
        mirror_min_depth_px=args.mirror_min_depth_px,
        strong_change_threshold=args.strong_change_threshold,
        weak_change_threshold=args.weak_change_threshold,
        weak_change_threshold_ratio=args.weak_change_threshold_ratio,
        weak_growth_steps=args.weak_growth_steps,
        approach_shadow_guard=args.approach_shadow_guard,
        min_strong_change_threshold=args.min_strong_change_threshold,
        min_component_area_px2=int(args.min_component_area_px2),
        morphology_radius_px=args.morphology_radius_px,
        contact_band_half_height_px=args.contact_band_half_height_px,
        rim_to_rim=bool(args.rim_to_rim),
        foreground_depth_px=args.foreground_depth_px,
        top_exclusion_ratio=args.top_exclusion_ratio,
        min_gas_area_px2=args.min_gas_area_px2,
        min_impact_width_px=args.min_impact_width_px,
        include_pre_impact=args.include_pre_impact,
        time_zero=args.time_zero,
        debug_dir=args.debug_dir or output_dir / "debug_overlays",
        debug_interval=args.debug_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        substrate=args.substrate,
        hydrophilicity=args.hydrophilicity,
        substrate_temp_deg_c=args.substrate_temp_deg_c,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    measurements = analyze_sequence(config)
    write_csv(config.output_csv, measurements)
    write_run_params(config.output_dir / RUN_PARAMS_FILENAME, config)

    impact_frame = measurements[0].impact_frame if measurements else None
    valid = [row for row in measurements if not math.isnan(row.d_spread_mm)]
    print(f"Wrote {len(measurements)} rows to {config.output_csv}")
    if valid:
        max_row = max(valid, key=lambda row: row.d_spread_mm)
        print(f"Surface y: {valid[0].surface_y_px} px")
        print(f"Impact frame: {impact_frame if impact_frame is not None else 'not detected'}")
        print(
            "Max spreading diameter: "
            f"{max_row.d_spread_mm:.6f} mm at {max_row.time_ms:.6f} ms"
        )
    else:
        print("No valid droplet diameter measurements were found.")
    if measurements and measurements[0].droplet_size_ref_frame is not None:
        ref = measurements[0]
        for method, diameter_mm in (
            ("volume", ref.ref_d_eq_volume_mm),
            ("area", ref.ref_d_eq_area_mm),
        ):
            if diameter_mm is not None:
                print(
                    f"Ref d_eq_{method}: {diameter_mm:.6f} mm "
                    f"at frame {ref.droplet_size_ref_frame}"
                )
