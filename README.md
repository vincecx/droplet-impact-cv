# droplet-impact-cv

Measure droplet spreading diameter from side-view high-speed TIFF/JPEG sequences
with a fully automatic CLI or a semi-automatic GUI.

## Quick start

Install the environment and dependencies with `uv`:

```bash
uv sync
```

Choose a mode:

| Mode | Use it to | Start command |
| --- | --- | --- |
| GUI | Preview frames, adjust analysis parameters, and process one folder. | `uv run droplet-impact-cv-gui` |
| CLI | Process a folder using saved settings and command-line options. | `uv run droplet-impact-cv path/to/image_sequences` |

The GUI requires Python with Tk support (`tkinter`). An experiment-specific CLI
command can supply custom settings:

```bash
uv run droplet-impact-cv path/to/image_sequences \
  --fps 1000 \
  --pixel-size-mm-per-px 0.001 \
  --surface-calibration-frame 61 \
  --reflection-mode mirror
```

## Prepare the input

Input images must be grayscale TIFF or 8-bit JPEG, with a six-digit frame number
at the end of each filename. For example, `capture_000005.tif` is reported as
frame 5 even when it is the first file in the input directory. The last six
digits are the frame number, including when the experiment name ends in a digit:
`example3000010.tif` is frame 10.

Folder names may contain spaces or non-ASCII characters. Quote the entire input
path when running from a shell, for example:

```powershell
uv run droplet-impact-cv "path/to/示例 example"
```

`--start-frame` and `--end-frame` restrict the working range using inclusive
filename frame numbers. The first frame in the selected working range should
be a clean background with no droplet outside of `--top-exclusion-ratio` range.

## GUI workflow

1. Use **Choose Folder...** to open one image folder and load its settings.
2. Inspect **Before Processing** and **After Processing**, then adjust the grouped analysis parameters. The initial preview shows the selected droplet size reference frame, or the first frame if Auto finds no reference.
3. Use **Save cv_config.txt** to persist the current settings if needed.
4. Use **Process All Frames** to analyze the configured working range with the current GUI settings and write the CSV and overlays.

### Preview and playback

Parameter changes process the selected frame and any required calibration or
threshold updates. Previewing does not rerun the full sequence or write outputs.
Use the frame slider and playback controls to inspect other frames, and the
mouse wheel over either preview to zoom around the cursor.

Playback speed defaults to 5 fps and is independent of camera `fps`, which sets
the measurement time scale. Playback waits for each frame to finish processing
if the requested speed cannot be reached.

### Frame selection and manual surface calibration

The substrate parameter group contains two independent frame controls:

| GUI control | Selects |
| --- | --- |
| **Droplet size reference frame** | The frame used for both reference equivalent diameters. Auto discovers a suitable frame. |
| **Surface calibration frame** | The impacted-droplet frame used to establish the fixed substrate line. Auto uses the clean background. |

**Select surface line manually** lets you select two points in the
**Before Processing** preview and confirm the line. This updates the surface
calibration frame and surface center y, turning off Auto for both. If surface
tilt is on Auto, a dialog lets you keep automatic tilt detection or turn off
Auto and use the selected line's tilt. Confirming reruns automatic calibration,
threshold calculation, and droplet size reference discovery, updating the Auto
values. Selection uses raw frames and remains available after preview errors.

### Save and reset settings

**Save cv_config.txt** writes the current analysis settings and output paths to
`<input-folder>/cv_config.txt`. The dialog offers three choices:

- **Yes:** rename the existing file to `cv_config.txt.bak`, replacing any previous backup, then save.
- **No:** overwrite without creating a backup.
- **Cancel:** leave the configuration file unchanged.

Auto settings remain automatic when reloaded. **Reset Parameters** restores the
values loaded when the folder was opened, including their Auto state.

## Configuration

Put experiment settings in `<input-folder>/cv_config.txt`. The file uses CLI
option syntax; blank lines and `#` comments are allowed:

```text
# Camera and scale
--fps 1000
--pixel-size-mm-per-px 0.01

# Fixed experiment metadata (does not affect measurements)
--substrate Substrate-name-1
--hydrophilicity SHO
--substrate_temp_deg_c 24

# Surface calibration
--surface-calibration-frame 10
--reflection-mode mirror
```

`--substrate`, `--hydrophilicity`, and `--substrate_temp_deg_c` record the surface
name, wettability (`SHI`, `HI`, `HO`, or `SHO`), and temperature in degrees Celsius.
They default to `null`, remain fixed during GUI preview and batch processing,
and are written under `parameters` in the output folder's `run_params.json`.
Quote metadata values containing spaces in `cv_config.txt`, such as
`--substrate "Surface 1"`. The GUI preserves this metadata when saving settings.
Use `--rim-to-rim {0,1}` to select the diameter measurement; the old
`--hydrophobic` option is rejected in both CLI arguments and configuration files.

Settings take precedence in this order, from highest to lowest:

| Mode | Precedence |
| --- | --- |
| CLI | Explicit command-line options > `cv_config.txt` > program defaults |
| GUI | Current GUI settings > `cv_config.txt` > program defaults |

The GUI accepts no command-line folder or analysis options; choose the folder
inside the application. For parameters with an Auto mode, omitting the option
from the configuration preserves its automatic behavior.

## Analysis and measurement

### Liquid detection

Liquid is detected by changes from the clean background, whether darker or
brighter. The main threshold identifies clear liquid; the faint threshold
extends the outline near already detected liquid. Enclosed bright regions are
filled as part of the drop.

For missing liquid edges, lower the main or faint thresholds, or increase faint
reach. For separate specks, increase minimum drop area or outline cleanup size.

### Substrate calibration

One substrate line is used for the whole sequence. The surface calibration
frame and `--reflection-mode` determine how that line is found:

| Reflection mode | Surface calibration |
| --- | --- |
| `mirror` | Uses the drop and its reflection. |
| `none` | Uses the drop's lower edge; suitable for weak reflections. |
| `auto` | Selects `mirror` only with strong reflection evidence; assumes `mirror` when no surface calibration frame is supplied. |

`--surface-y-px` overrides the line's center position. Surface angles are
clockwise; without calibration or an explicit angle, the fallback is `0` deg.

### Spreading diameter, impact, and time

`d_spread` is the outermost liquid span within the substrate band
(`--surface-band-px`, default 1 px); it is zero when the drop is clear of that
band. With `--rim-to-rim 1`, it is the full rim-to-rim width instead.

Impact is always detected from contact-line width reaching
`--min-impact-width-px`. By default, time is zero at impact and pre-impact rows
are omitted. `--include-pre-impact` retains those rows, and
`--time-zero first-frame` changes the time origin.

### Reference equivalent diameters

`D_parallel` and `D_normal` are the full silhouette spans parallel and normal
to the substrate. Both reference equivalent diameters use the same frame:

- **Auto:** the first frame whose lower-contour distance to the substrate is at most `D_eq_volume / 2`.
- **Explicit:** the frame selected by `--droplet-size-ref-frame` or the GUI control. It must exist in the working range and contain a measurable droplet; it bypasses the automatic clearance condition.

Selecting the size reference frame does not change the surface calibration.
The two equivalent diameters are calculated as follows:

| Quantity | Formula | Interpretation |
| --- | --- | --- |
| `ref_d_eq_volume` | `(D_parallel² × D_normal)^(1/3)` | Equivalent sphere diameter of an axisymmetric ellipsoid with the measured spans. |
| `ref_d_eq_area` | `sqrt(4 × A / pi)` | Equivalent circle diameter of the measured side-view silhouette area `A`. |

Both methods use the filled silhouette before surface clipping. The area method
measures a 2D projection, not volume. Reference dimensions and clearance are
frozen at `droplet_size_ref_frame` and repeated on every CSV row. If Auto
finds no reference, those columns are empty.

## Outputs

### Output locations

The default CSV filename is `spreading_diameter.csv`.

| Mode / GUI output choice | Output folder |
| --- | --- |
| CLI default / GUI: Project outputs | `<project-dir>/outputs/<input-folder-name>/` |
| GUI: Input folder (default) | `<input-folder>/CV_postprocessing/` |
| GUI: Custom folder | Selected directory |

`<project-dir>` is the directory containing `pyproject.toml`; the automatic
default does not depend on the input location or current working directory.
CLI output paths can be overridden with `--output-csv` and `--debug-dir`.

GUI output selection overrides configured output paths, preserves the configured
CSV filename, and puts overlays in `<output-folder>/debug_overlays/`. For example,
the input-folder choice for `sourcedata/example` writes to
`sourcedata/example/CV_postprocessing/`.

### Diagnostic overlays

Overlays are written every frame by default; `--debug-interval` controls the
regular interval. The first frame, size reference frame, and frames near impact
can also produce overlays. Each full run first removes existing frame-numbered
overlay PNGs from the target debug directory.

### CSV columns and naming

Python and CSV names use lowercase snake_case; CLI names use corresponding
hyphens. `d` means diameter, `eq` means equivalent, and `ref_` marks quantities
frozen at the size reference frame. Unit suffixes are `_px`, `_mm`, `_ms`, `_deg`,
and `_px2` for area. Pixel size uses `_mm_per_px`; `fps` is the camera frame rate.

| Columns | Meaning |
| --- | --- |
| `filename`, `frame_number`, `time_ms` | Source file, filename frame number, and time. |
| `d_spread_px`, `d_spread_mm` | Spreading diameter selected by `--rim-to-rim`. |
| `d_contact_px`, `d_contact_mm` | Surface contact-line diameter. |
| `d_rim_px`, `d_rim_mm` | Rim-to-rim diameter, available in rim-to-rim mode. |
| `droplet_size_ref_frame` | Frame used for both reference equivalent diameters. |
| `ref_d_eq_volume_px`, `ref_d_eq_volume_mm` | Volume-equivalent reference diameter. |
| `ref_d_eq_area_px`, `ref_d_eq_area_mm` | Area-equivalent reference diameter. |
| `ref_d_parallel_px`, `ref_d_parallel_mm` | Reference projection span parallel to the substrate. |
| `ref_d_normal_px`, `ref_d_normal_mm` | Reference projection span normal to the substrate. |
| `ref_clearance_px`, `ref_clearance_mm` | Reference lower-contour clearance normal to the substrate. |
| `impact_frame`, `surface_y_px` | Detected impact frame and fixed surface-line center y coordinate. |
| `droplet_area_px2` | Current frame's filled gas-side area after surface clipping. |
| `fps`, `pixel_size_mm_per_px`, `surface_calibration_frame` | Camera and surface calibration metadata. |

## Parameter reference

Analysis groups match the GUI. Input, impact/time, and output options have
additional functional groups. Parameters within each group are sorted
alphabetically by name, ignoring leading dashes.

### Experiment and scale

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--fps` | `1000` | Camera frame rate used to calculate `time_ms`. |
| `--pixel-size-mm-per-px` | `0.01` | Physical length represented by one pixel. |

### Substrate and width measurement

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--droplet-size-ref-frame` | automatic | Frame used for both reference equivalent diameters; Auto uses the first frame whose substrate clearance is at most half its volume-equivalent diameter. |
| `--rim-to-rim {0,1}` | `0` | Set to `1` to measure the silhouette's rim-to-rim width instead of its contact-line width. |
| `--substrate` | `None` | Surface name; experiment metadata only. |
| `--hydrophilicity {SHI,HI,HO,SHO}` | `None` | Surface wettability; experiment metadata only. |
| `--substrate_temp_deg_c` | `None` | Surface temperature in degrees Celsius; experiment metadata only. |
| `--mirror-min-depth-px` | `50` | Minimum mirror candidate depth below the coarse surface line; also used for automatic reflection classification. |
| `--reflection-mode {auto,mirror,none}` | `auto` | Select automatic, reflection-symmetry, or lower-contour surface calibration. |
| `--surface-angle-deg` | detected or `0` | Manually override the clockwise surface angle. |
| `--surface-band-px` | `1` | Half-height of the band in which spreading width is measured. |
| `--surface-calibration-frame` | none | Impacted-droplet frame used to calibrate the fixed surface line. |
| `--surface-y-px` | automatic | Manually override the surface-line center y coordinate. |

### Liquid detection

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--foreground-depth-px` | `220` | Search depth below the substrate line. |
| `--min-strong-threshold` | `1500` | Lower bound used by automatic strong-threshold estimation. |
| `--strong-threshold` | automatic | Main liquid-detection threshold; lower values include fainter liquid and more noise. |
| `--top-exclusion-ratio` | `0.25` | Fraction of image height excluded from droplet detection at the top, between 0 and 1. Background and surface calibration still use the full image. |
| `--weak-reach-px` | `40` | Maximum distance faint liquid can extend from clearly detected liquid. |
| `--weak-threshold` | ratio-derived | Explicit weak threshold for faint liquid. |
| `--weak-threshold-ratio` | `0.3` | Weak/strong threshold ratio when no explicit weak threshold is set. |

### Drop selection and outline cleanup

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--min-contact-px` | `30` | Minimum detected pixels above the substrate for a selected drop. |
| `--min-droplet-area-px2` | `250` | Ignore detected objects smaller than this area. |
| `--morphology-radius-px` | `3` | Cleanup size for closing gaps and removing small specks. |

### Input and working range

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--end-frame` | unbounded | Inclusive last filename frame number in the working range. |
| `input_dir` | `sourcedata/example` | Folder containing numbered TIFF/JPEG frames and optional `cv_config.txt`. |
| `--start-frame` | unbounded | Inclusive first filename frame number; this frame becomes the clean background. |

### Impact and time

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--include-pre-impact`, `--no-include-pre-impact` | disabled | Include or exclude rows before impact. |
| `--min-impact-width-px` | `30` | Contact-line diameter that marks the first impact frame. |
| `--time-zero {impact,first-frame}` | `impact` | Choose the time origin. |

### Output folder and diagnostics

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--debug-dir` | `<project-dir>/outputs/<input-folder-name>/debug_overlays` | Diagnostic overlay directory. |
| `--debug-interval` | `1` | Write one debug overlay every N frames. |
| `--output-csv`, `-o` | `<project-dir>/outputs/<input-folder-name>/spreading_diameter.csv` | CSV output path. |

## Development

Run the test suite with:

```bash
uv run python -m unittest discover -v
```
