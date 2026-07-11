# droplet-impact-cv

Automated side-view image analysis for droplet impact sequences. The CLI reads
grayscale TIFF frames or 8-bit JPEG exports, detects the substrate surface and
impact frame, then exports droplet spreading diameter as a function of time.

Each image filename must end with a six-digit frame number immediately before the
extension. For example, `capture_000005.tif` is reported as frame 5 even when it
is the first file in the input directory.

The frame with the smallest frame number in the input folder is assumed to be a
clean background image containing no droplet and is used directly as the
background for the full sequence.

## Usage

The project uses `uv` for dependency and environment management.

```bash
uv run droplet-impact-cv sourcedata/example
```

Default physical parameters:

- frame rate: `8000` fps
- pixel size: `0.00711883341` mm/px

Common overrides:

```bash
uv run droplet-impact-cv path/to/tiff_frames \
  --fps 8000 \
  --pixel-size-mm 0.00711883341 \
  --surface-frame 61 \
  -o outputs/example/spreading_diameter.csv \
  --debug-every 1
```

Use `--start-frame 20 --end-frame 120` to process only files whose six-digit
filename frame number is in the inclusive range 20 through 120. Either option
can be omitted to leave that side of the range unbounded. When `--start-frame`
is specified, the first frame in the selected range is used as the background.

The program automatically reads command-line-style options from
`<input-folder>/cv_config.txt` when that file exists. Blank lines and comments
starting with `#` are allowed. For example:

```text
--fps 4000
--pixel-size-mm 0.01682736321
--surface-frame 9105
--reflection-mode mirror
```

Configuration precedence is: explicit command-line options, then
`cv_config.txt`, then the defaults defined by the program. For example,
`--fps 8000` on the command line overrides `--fps 4000` in `cv_config.txt`.

Use `--surface-frame` to select a frame containing an impacted droplet, then use
`--reflection-mode` to choose how the surface line is calibrated from it. The
calibrated line is the single fixed surface reference for every frame.

- `mirror`: fit the surface through the contact vertices between the droplet and
  its strong reflection.
- `none`: fit the surface to the lower contour of the impacted droplet in the
  selected surface frame; weak reflections are treated as `none`.
- `auto` (default): use `mirror` only when the calibration silhouette provides
  strong reflection evidence; otherwise use `none`.

Automatic classification is deliberately conservative. Set the mode explicitly
in `cv_config.txt` or on the command line for transparent droplets, weak
reflections, or backgrounds with multiple horizontal edges. Command-line values
override the per-folder configuration as usual.

Surface angles are measured clockwise from horizontal. Use
`--surface-angle-deg` to override the automatically detected angle. If no
`--surface-frame` or explicit angle is provided, the fallback is `-0.6` degrees
(that is, `0.6` degrees counterclockwise). Use `--surface-y` to manually override
the calibrated surface-line center pixel coordinate.

## Foreground detection

The droplet is not assumed to be dark. A pixel counts as foreground when its
gray level differs from the background frame in either direction, so a
transparent droplet that refracts the backlight and appears brighter than a
dark substrate is still detected.

Detection uses two thresholds. Pixels whose absolute change exceeds
`--threshold` (default: automatic, at least `--min-foreground-delta`) are
reliable foreground. Pixels above the weak threshold (default: a quarter of
the effective strong threshold, override with `--weak-threshold`) are included
only when they are connected to reliable foreground within a bounded distance.
This recovers the faint interior of a nearly transparent liquid film while
rejecting isolated weak artifacts such as diffuse out-of-focus shadows.

## Outputs

By default, all outputs are grouped under a folder named after the input
folder. e.g., For the command `uv run droplet-impact-cv sourcedata/example`, the
CSV is written to `outputs/example/spreading_diameter.csv` and diagnostic
overlays are written to `outputs/example/debug_overlays`. Use `--debug-dir` to
choose a different directory for debug overlays.

The reported spreading diameter is the maximum width of the detected liquid
silhouette within 3 px above or below the fixed surface line: the distance
between the outermost points where the silhouette enters that band. Frames
where the silhouette stays clear of the band report a diameter of `0`.

Debug overlays are written selectively rather than for every frame. Each run
first deletes any frame-numbered overlay PNGs already in the target directory,
so its contents always reflect a single run's calibration.

The CSV columns are:

- `filename`
- `frame`
- `time_ms`
- `diameter_px`
- `diameter_mm`
- `impact_frame`
- `surface_y_px`
- `component_area_px`
- `fps`
- `pixel-size-mm`
- `surface-frame`

By default, `time_ms = 0` at the automatically detected impact frame and
pre-impact rows are omitted. Use `--include-pre-impact` to keep all frames, or
`--time-zero first-frame` to report time from the first image.

## Project structure

- `droplet_impact_cv/cli.py`: command-line parsing and entry point
- `droplet_impact_cv/models.py`: configuration and result data models
- `droplet_impact_cv/imaging.py`: TIFF loading, segmentation, and surface calibration
- `droplet_impact_cv/analysis.py`: sequence-level analysis workflow
- `droplet_impact_cv/visualization.py`: debug overlay rendering
- `droplet_impact_cv/output.py`: CSV serialization
- `tests/`: unit tests grouped by the module under test

## Development

Synchronize the environment and run the test suite with:

```bash
uv sync
uv run python -m unittest discover -v
```
