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
  --pixel-size-mm-per-px 0.00711883341 \
  --calibration-frame 61 \
  --reflection-mode auto
```

### Semi-automatic GUI mode

```bash
uv run droplet-impact-cv-gui
```

The fully automatic CLI remains available as `droplet-impact-cv`. The GUI uses
English labels and accepts no input-directory argument or analysis overrides.
Choose exactly one image folder with **Choose Folder...**. Initial settings
come from that folder's `cv_config.txt`, then program defaults; changes in the
GUI take precedence for the current session only.

On opening the folder, the GUI loads the background and calibration, then scans
only until it finds the first `ref_frame`. That frame is the default preview;
if no reference is detected, the first frame is shown. The frame slider includes
only existing images inside the configured `start_frame` / `end_frame` range.
The first selected frame remains the background, as in automatic mode.

**Before Processing** and **After Processing** show the same frame side by side.
Adjust contour parameters using sliders or number entries. Each control includes
a short explanation and the expected adjustment effect. **Auto** preserves the
automatic strong/weak threshold or surface calibration logic; clear it to use an
explicit value. Suggested slider ranges expand to accommodate larger configured
or typed values. Reflection mode and hydrophobic measurement are also adjustable.
**Reset Parameters** restores settings loaded from the folder and defaults.

Edits and frame changes refresh only the selected frame after a short debounce.
Background and file selection are reused. If a parameter affects calibration or
automatic threshold estimation, those supporting calculations are refreshed;
the full sequence and reference search are not rerun during preview. Previewing
creates no output files and does not clear existing debug overlays.

Click **Process All Frames** to analyze the complete working range with a snapshot
of the current GUI settings. Controls are locked during processing and a progress
bar shows completion. Choose an **Output Folder** using one of three options:

- **Project outputs (same as automatic mode)**: `<project-dir>/outputs/<input-folder-name>/`.
- **Input folder / CV_postprocessing**: `<input-folder>/CV_postprocessing/`.
- **Custom folder**: enter a directory or use **Browse...** to select one.

The selected folder contains the CSV, and diagnostic images always go into its
`debug_overlays` subfolder. The configured CSV filename is preserved. A configured
CSV path outside the automatic output directory initially selects **Custom folder**;
otherwise the automatic location is selected. The GUI output choice overrides
configured output paths, including `--debug-dir`.

Filtering and measurement rules match automatic mode. Other settings, including
fps, pixel size, calibration frame, and pre-impact export, remain as loaded from
`cv_config.txt` or defaults. Parameters are never written back to `cv_config.txt`.

The GUI requires Python with Tk support (`tkinter`); no additional GUI package
is required by the project's Windows uv environment.

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
5. Clean the foreground mask, select the gas-side droplet component nearest the
   surface, fill its enclosed interior, and measure the surface-contact span.
6. Detect impact when the contact-line diameter reaches
   `--min-impact-width-px`, convert px to mm and frames to time,
   then write the CSV and selected overlays.
7. Before impact, define the reference drop size at the first frame where the
   surface-normal clearance of the lower silhouette reaches one half of its
   equivalent diameter.

The detected foreground therefore represents image change. Transparent liquid 
that brightens a dark substrate can still be part of the droplet silhouette.

Surface calibration uses `mirror` to fit the contact vertices between a droplet
and its strong reflection, or `none` to fit the impacted droplet's lower
contour. The default `auto` mode conservatively selects `mirror` only when the
silhouette provides strong reflection evidence. Setting the mode explicitly is 
recommended.

Surface angles are clockwise from horizontal. Without a calibration frame or
explicit `--surface-angle-deg`, the fallback is `-0.6` deg. A manual
`--surface-y-px` overrides the calibrated surface-line center.

### CLI parameter summary

Options can also be placed in `<input-folder>/cv_config.txt`, one or more per
line. Blank lines and `#` comments are allowed. Precedence is: explicit CLI
options, `cv_config.txt`, then program defaults.

Example of cv_config.txt:
```text
--fps 4000
--pixel-size-mm-per-px 0.01682736321
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
| `--pixel-size-mm-per-px` | `0.00711883341` | Physical length represented by one pixel. |
| `--calibration-frame` | none | Impacted-droplet frame used to calibrate the fixed surface line. |
| `--reflection-mode {auto,mirror,none}` | `auto` | Select automatic, reflection-symmetry, or lower-contour surface calibration. |
| `--surface-y-px` | automatic | Manually override the surface-line center y coordinate. |
| `--surface-angle-deg` | detected or `-0.6` | Manually override the clockwise surface angle. |
| `--strong-threshold` | automatic | Strong absolute background-change threshold. |
| `--min-strong-threshold` | `1500` | Lower bound used by automatic strong-threshold estimation. |
| `--weak-threshold` | ratio-derived | Explicit weak threshold for faint liquid. |
| `--weak-threshold-ratio` | `0.25` | Weak/strong threshold ratio when no explicit weak threshold is set. |
| `--weak-reach-px` | `40` | Maximum distance weak foreground may grow from strong foreground. |
| `--min-droplet-area-px2` | `250` | Reject connected components smaller than this pixel count. |
| `--morphology-radius-px` | `3` | Radius used to close/fill the mask and remove small artifacts. |
| `--surface-band-px` | `3` | Half-height of the band in which spreading width is measured. |
| `--hydrophobic {0,1}` | `0` | Set to `1` for hydrophobic surfaces to measure the silhouette's rim-to-rim width instead of its contact-line width. |

Threshold tuning usually starts with `--strong-threshold`. For a faint
transparent interior, lower `--weak-threshold-ratio` or set `--weak-threshold`
directly; keep `--weak-reach-px` only as large as needed to bridge the liquid
interior. If noise forms separate components, tune `--min-droplet-area-px2` and
`--morphology-radius-px` instead.

#### Other parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `-o`, `--output-csv` | `<project-dir>/outputs/<input-folder-name>/spreading_diameter.csv` | CSV output path. |
| `--foreground-depth-px` | `220` | Foreground ROI depth below the surface. |
| `--min-contact-px` | `30` | Minimum gas-side foreground px for the selected droplet. |
| `--min-impact-width-px` | `30` | Contact-line diameter that marks the first impact frame. |
| `--include-pre-impact`, `--no-include-pre-impact` | disabled | Include or exclude rows before impact. |
| `--time-zero {impact,first-frame}` | `impact` | Choose the time origin. |
| `--debug-dir` | `<project-dir>/outputs/<input-folder-name>/debug_overlays` | Diagnostic overlay directory. |
| `--debug-interval` | `1` | Write one debug overlay every N frames. |

## Outputs

In fully automatic mode, default outputs are anchored to the project directory
(the directory containing `pyproject.toml`), independently of the input location
and current working directory. For `uv run droplet-impact-cv sourcedata/example`,
the CSV goes to `<project-dir>/outputs/example/spreading_diameter.csv` and overlays
to `<project-dir>/outputs/example/debug_overlays`. Use `--output-csv` and
`--debug-dir` to override these automatic-mode paths.

In GUI mode, the input-folder shortcut for `sourcedata/example` writes the CSV to
`sourcedata/example/CV_postprocessing/spreading_diameter.csv` and overlays to
`sourcedata/example/CV_postprocessing/debug_overlays/`. Both shortcuts and manual
selection keep overlays under `debug_overlays` inside the selected output folder.

The reported `d_spread` is the maximum width of the
detected liquid silhouette within `--surface-band-px` px above or below the
fixed surface line (default: 3): the distance between the outermost points
where the silhouette enters that band. Frames where the silhouette stays clear
of the band report a spreading diameter of `0`. With `--hydrophobic 1`, it is instead the
rim-to-rim width of the entire detected droplet silhouette, so lifted droplets
remain measurable. Regardless of `--hydrophobic`, a frame is marked as the impact frame
once its contact-line diameter reaches `--min-impact-width-px` (default: 30).

### Reference equivalent diameters

The program uses substrate-aligned coordinates: `parallel` is tangent to the
fixed surface line, while `normal` is perpendicular to it. For each selected
silhouette it measures the full projection spans `D_parallel` and `D_normal`.
The first frame satisfying `surface_clearance ≤ D_eq_volume / 2` is saved as
`ref_frame`, using the volume-equivalent method for selection.

Both equivalent diameters are measured in that same reference frame:

- `ref_d_eq_volume`: `(D_parallel² × D_normal)^(1/3)`, the equivalent sphere diameter of an axisymmetric ellipsoid with the measured spans.
- `ref_d_eq_area`: `sqrt(4 × A / pi)`, the equivalent circle diameter of the measured side-view silhouette area `A`.

The area counts px inside the selected, filled droplet contour, including its
enclosed bright regions. Both methods use the silhouette before surface
clipping. The area method describes a 2D projection; it does not infer volume.
Each diameter is exported in px and mm and printed in mm in the CLI summary.
The reference spans and clearance are also frozen at `ref_frame`, and
all reference values repeat on every exported row. If no reference frame is
found, reference columns are empty.

Overlays show the source image, fixed surface line, detected mask, and available
contact/rim measurement lines. Labels use `d_contact` and `d_rim`.

By default, debug overlays are generated for every frame (use `--debug-interval`
to change). Note that each run first deletes any frame-numbered overlay PNGs
already in the target directory, so its contents always reflect a single run's
calibration.

Parameter names use lowercase snake_case in Python and CSV, and corresponding
hyphenated names in the CLI. `d` means diameter and `eq` means equivalent;
`ref_` marks quantities frozen at the reference frame. Unit suffixes are
`_px`, `_mm`, `_ms`, `_deg`, and `_px2` for area. Pixel size uses `_mm_per_px`;
`fps` is the camera frame rate.

The CSV columns are:

| Columns | Meaning |
| --- | --- |
| `filename`, `frame_number`, `time_ms` | Source file, filename frame number, and time. |
| `d_spread_px`, `d_spread_mm` | Spreading diameter selected by `--hydrophobic`. |
| `d_contact_px`, `d_contact_mm` | Surface contact-line diameter. |
| `d_rim_px`, `d_rim_mm` | Rim-to-rim diameter, available in hydrophobic mode. |
| `ref_frame` | Frame used for both reference equivalent diameters. |
| `ref_d_eq_volume_px`, `ref_d_eq_volume_mm` | Volume-equivalent reference diameter. |
| `ref_d_eq_area_px`, `ref_d_eq_area_mm` | Area-equivalent reference diameter. |
| `ref_d_parallel_px`, `ref_d_parallel_mm` | Reference projection span parallel to the substrate. |
| `ref_d_normal_px`, `ref_d_normal_mm` | Reference projection span normal to the substrate. |
| `ref_clearance_px`, `ref_clearance_mm` | Reference lower-contour clearance normal to the substrate. |
| `impact_frame`, `surface_y_px` | Detected impact frame and fixed surface-line center y coordinate. |
| `droplet_area_px2` | Current frame's filled gas-side area after surface clipping. |
| `fps`, `pixel_size_mm_per_px`, `calibration_frame` | Camera and surface calibration metadata. |

By default, `time_ms = 0` at the automatically detected impact frame and
pre-impact rows are omitted. Use `--include-pre-impact` to keep all frames, or
`--time-zero first-frame` to report time from the first image.

## Project structure

- `droplet_impact_cv/cli.py`: fully automatic command-line parsing and entry point
- `droplet_impact_cv/gui.py`: semi-automatic GUI, parameter controls, and single-frame preview
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
