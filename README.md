# droplet-impact-cv

Automated image analysis for side-view high-speed droplet impact sequences. 
The CLI reads grayscale TIFF or 8-bit JPEG sequences, detects the substrate
surface and impact frame, then exports droplet spreading diameter as a function
of time.

Each image filename must end with a six-digit frame number immediately before the
extension. For example, `capture_000005.tif` is reported as frame 5 even when it
is the first file in the input directory.

The frame with the smallest frame number in the input folder is assumed to be a
clean background image containing no droplet and is used directly as the
background for the full sequence.

## Usage

The project uses `uv` for dependency and environment management.

```bash
uv sync
uv run droplet-impact-cv path/to/image_sequences
```

An experiment-specific command typically looks like this:

```bash
uv run droplet-impact-cv path/to/image_sequences \
  --fps 8000 \
  --pixel-size-mm 0.00711883341 \
  --calibration-frame 61 \
  --reflection-mode auto
```

### Runtime logic

1. Read the TIFF/JPEG frames whose filenames end in a six-digit frame number,
   then optionally restrict them with `--start-frame` and `--end-frame`.
2. Use the first selected frame as the clean background. Detect a coarse
   substrate edge from that image.
3. If `--calibration-frame` is set, calibrate one fixed surface line from the
   impacted droplet in that frame. `--reflection-mode` controls whether the
   droplet reflection participates in this calibration.
4. Subtract each frame from the background in both intensity directions. A
   strong threshold identifies reliable foreground; a connected weak threshold
   recovers faint, transparent liquid without admitting isolated weak artifacts.
5. Clean the foreground mask, select the component contacting the surface, fill
   its enclosed interior, and measure its outermost span inside the surface band.
6. Detect impact when that span reaches `--min-impact-width-px`, convert pixels
   to millimetres and frames to time, then write the CSV and selected overlays.

The detected foreground therefore represents image change. Transparent liquid 
that brightens a dark substrate can still be part of the droplet silhouette.

Surface calibration uses `mirror` to fit the contact vertices between a droplet
and its strong reflection, or `none` to fit the impacted droplet's lower
contour. The default `auto` mode conservatively selects `mirror` only when the
silhouette provides strong reflection evidence. Setting the mode explicitly is 
recommended.

Surface angles are clockwise from horizontal. Without a calibration frame or
explicit `--surface-angle-deg`, the fallback is `-0.6` degrees. A manual
`--surface-y-px` overrides the calibrated surface-line center.

### CLI parameter summary

Options can also be placed in `<input-folder>/cv_config.txt`, one or more per
line. Blank lines and `#` comments are allowed. Precedence is: explicit CLI
options, `cv_config.txt`, then program defaults.

Example of cv_config.txt:
```text
--fps 4000
--pixel-size-mm 0.01682736321
--calibration-frame 9105
--reflection-mode mirror
```

The names below are the names shown by `--help` and accepted by the CLI.

#### Key parameters

These cover input selection, physical calibration, surface calibration, and the
settings most useful when tuning the detected droplet contour.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `input_dir` | `sourcedata/example` | Folder containing numbered TIFF/JPEG frames and optional `cv_config.txt`. |
| `--start-frame`, `--end-frame` | unbounded | Inclusive filename frame-number range; the first selected frame becomes the background. |
| `--fps` | `8000` | Camera frame rate used to calculate `time_ms`. |
| `--pixel-size-mm` | `0.00711883341` | Physical length represented by one pixel. |
| `--calibration-frame` | none | Impacted-droplet frame used to calibrate the fixed surface line. |
| `--reflection-mode {auto,mirror,none}` | `auto` | Select automatic, reflection-symmetry, or lower-contour surface calibration. |
| `--surface-y-px` | automatic | Manually override the surface-line center y coordinate. |
| `--surface-angle-deg` | detected or `-0.6` | Manually override the clockwise surface angle. |
| `--strong-threshold` | automatic | Strong absolute background-change threshold. |
| `--min-strong-threshold` | `1500` | Lower bound used by automatic strong-threshold estimation. |
| `--weak-threshold` | ratio-derived | Explicit weak threshold for faint liquid. |
| `--weak-threshold-ratio` | `0.25` | Weak/strong threshold ratio when no explicit weak threshold is set. |
| `--weak-reach-px` | `40` | Maximum distance weak foreground may grow from strong foreground. |
| `--min-droplet-area-px` | `250` | Reject connected components smaller than this pixel count. |
| `--morphology-radius-px` | `3` | Radius used to close/fill the mask and remove small artifacts. |
| `--surface-band-px` | `3` | Half-height of the band in which spreading width is measured. |
| `--HO {0,1}` | `0` | Set to `1` for hydrophobic surfaces to measure the silhouette's rim-to-rim width instead of its contact-line width. |

Threshold tuning usually starts with `--strong-threshold`. For a faint
transparent interior, lower `--weak-threshold-ratio` or set `--weak-threshold`
directly; keep `--weak-reach-px` only as large as needed to bridge the liquid
interior. If noise forms separate components, tune `--min-droplet-area-px` and
`--morphology-radius-px` instead.

#### Other parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `-o`, `--output-csv` | `outputs/<input-folder>/spreading_diameter.csv` | CSV output path. |
| `--foreground-depth-px` | `220` | Foreground ROI depth below the surface. |
| `--contact-above-px` | `20` | Component-selection window above the surface. |
| `--contact-below-px` | `170` | Component-selection window below the surface. |
| `--min-contact-pixels` | `30` | Required foreground pixel count just above the surface. |
| `--min-impact-width-px` | `30` | Width that marks the first impact frame. |
| `--include-pre-impact`, `--no-include-pre-impact` | disabled | Include or exclude rows before impact. |
| `--time-zero {impact,first-frame}` | `impact` | Choose the time origin. |
| `--debug-dir` | `outputs/<input-folder>/debug_overlays` | Diagnostic overlay directory. |
| `--debug-interval` | `1` | Write one debug overlay every N frames. |

## Outputs

By default, all outputs are grouped under a folder named after the input
folder. e.g., For the command `uv run droplet-impact-cv sourcedata/example`, the
CSV is written to `outputs/example/spreading_diameter.csv` and diagnostic
overlays are written to `outputs/example/debug_overlays`. Use `--debug-dir` to
choose a different directory for debug overlays.

By default, the reported spreading diameter is the maximum width of the
detected liquid silhouette within `--surface-band-px` pixels above or below the
fixed surface line (default: 3): the distance between the outermost points
where the silhouette enters that band. Frames where the silhouette stays clear
of the band report a diameter of `0`. With `--HO 1`, it is instead the
rim-to-rim width of the entire detected droplet silhouette, so lifted droplets
remain measurable. A frame is marked as the impact frame once the selected
width reaches `--min-impact-width-px` (default: 30).

By default, debug overlays are generated for every frame (use `--debug-interval`
to change). Note that each run first deletes any frame-numbered overlay PNGs
already in the target directory, so its contents always reflect a single run's
calibration.

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

Run the test suite with:

```bash
uv run python -m unittest discover -v
```
