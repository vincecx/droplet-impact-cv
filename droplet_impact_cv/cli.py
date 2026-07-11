from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

from .analysis import analyze_sequence
from .models import (
    DEFAULT_FPS,
    DEFAULT_MIN_FOREGROUND_DELTA,
    DEFAULT_PIXEL_SIZE_MM,
    REFLECTION_MODES,
    AnalysisConfig,
)
from .output import write_csv


CONFIG_FILENAME = "cv_config.txt"


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


def default_output_dir(input_dir: Path) -> Path:
    return Path("outputs") / input_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
        "--output",
        dest="output_csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: outputs/<input-folder>/spreading_diameter.csv",
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
        help="Camera frame rate in fps. Default: 8000",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=positive_float,
        default=DEFAULT_PIXEL_SIZE_MM,
        help="Physical length per pixel in mm. Default: 0.00711883341",
    )
    parser.add_argument(
        "--surface-y",
        type=nonnegative_int,
        default=None,
        help="Override detected surface y coordinate in pixels.",
    )
    parser.add_argument(
        "--surface-frame",
        type=nonnegative_int,
        default=None,
        help=(
            "Frame number containing an impacted droplet, used to calibrate "
            "the surface line according to --reflection-mode."
        ),
    )
    parser.add_argument(
        "--reflection-mode",
        choices=REFLECTION_MODES,
        default="auto",
        help=(
            "Surface calibration and measurement mode: auto, mirror, or none. "
            "Weak reflections should use none. Default: auto."
        ),
    )
    parser.add_argument(
        "--surface-angle-deg",
        type=float,
        default=None,
        help=(
            "Override surface angle in clockwise degrees. By default the angle is "
            "detected from --surface-frame, or -0.6 when no calibration frame is given."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=positive_float,
        default=None,
        help="Override foreground change threshold in gray levels.",
    )
    parser.add_argument(
        "--weak-threshold",
        type=positive_float,
        default=None,
        help=(
            "Weak threshold in gray levels for extending detections across "
            "faint transparent liquid. Pixels above this value count as "
            "foreground only when connected to a pixel above --threshold. "
            "Default: a quarter of the effective threshold."
        ),
    )
    parser.add_argument(
        "--min-foreground-delta",
        type=positive_float,
        default=DEFAULT_MIN_FOREGROUND_DELTA,
        help="Lower bound for automatic threshold. Default: 1500",
    )
    parser.add_argument(
        "--min-area-px",
        type=positive_float,
        default=250,
        help="Minimum liquid component area in pixels.",
    )
    parser.add_argument(
        "--morphology-radius-px",
        type=nonnegative_int,
        default=3,
        help="Morphology radius for mask cleanup.",
    )
    parser.add_argument(
        "--measure-above-surface-px",
        type=nonnegative_int,
        default=20,
        help="Measurement window above detected surface.",
    )
    parser.add_argument(
        "--measure-below-surface-px",
        type=nonnegative_int,
        default=220,
        help="Measurement window below detected surface.",
    )
    parser.add_argument(
        "--touch-above-surface-px",
        type=nonnegative_int,
        default=20,
        help="Impact detection window above surface.",
    )
    parser.add_argument(
        "--touch-below-surface-px",
        type=nonnegative_int,
        default=170,
        help="Impact detection window below surface.",
    )
    parser.add_argument(
        "--min-touch-pixels",
        type=nonnegative_int,
        default=30,
        help="Minimum foreground pixels near surface to mark impact.",
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
        help="Directory for overlay PNG diagnostics. Default: outputs/<input-folder>/debug_overlays",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=25,
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
    output_dir = default_output_dir(args.input_dir)
    return AnalysisConfig(
        input_dir=args.input_dir,
        output_csv=args.output_csv or output_dir / "spreading_diameter.csv",
        fps=args.fps,
        pixel_size_mm=args.pixel_size_mm,
        surface_y=args.surface_y,
        surface_frame=args.surface_frame,
        surface_angle_deg=args.surface_angle_deg,
        reflection_mode=args.reflection_mode,
        threshold=args.threshold,
        weak_threshold=args.weak_threshold,
        min_foreground_delta=args.min_foreground_delta,
        min_area_px=int(args.min_area_px),
        morphology_radius_px=args.morphology_radius_px,
        measure_above_surface_px=args.measure_above_surface_px,
        measure_below_surface_px=args.measure_below_surface_px,
        touch_above_surface_px=args.touch_above_surface_px,
        touch_below_surface_px=args.touch_below_surface_px,
        min_touch_pixels=args.min_touch_pixels,
        include_pre_impact=args.include_pre_impact,
        time_zero=args.time_zero,
        debug_dir=args.debug_dir or output_dir / "debug_overlays",
        debug_every=args.debug_every,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    measurements = analyze_sequence(config)
    write_csv(config.output_csv, measurements)

    impact_frame = measurements[0].impact_frame if measurements else None
    valid = [row for row in measurements if not math.isnan(row.diameter_mm)]
    print(f"Wrote {len(measurements)} rows to {config.output_csv}")
    if valid:
        max_row = max(valid, key=lambda row: row.diameter_mm)
        print(f"Surface y: {valid[0].surface_y}px")
        print(f"Impact frame: {impact_frame if impact_frame is not None else 'not detected'}")
        print(f"Max diameter: {max_row.diameter_mm:.6f} mm at {max_row.time_ms:.6f} ms")
    else:
        print("No valid droplet diameter measurements were found.")
