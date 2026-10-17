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
  --surface-calibration-method mirror
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
--surface-calibration-method mirror
```

`--substrate`, `--hydrophilicity`, and `--substrate_temp_deg_c` record the surface
name, wettability (`SHI`, `HI`, `HO`, or `SHO`), and temperature in degrees Celsius.
They default to `null`, remain fixed during GUI preview and batch processing,
and are written under `parameters` in the output folder's `run_params.json`.
Quote metadata values containing spaces in `cv_config.txt`, such as
`--substrate "Surface 1"`. The GUI preserves this metadata when saving settings.
Use `--rim-to-rim {0,1}` to select the diameter measurement.

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
brighter. The strong change threshold identifies clear liquid; the weak change threshold
extends the outline near already detected liquid. Enclosed bright regions are
filled as part of the drop.

Near-surface shadow suppression is enabled by default. Two preceding complete
elliptical main-threshold silhouettes must agree in size and motion with the
approaching drop. The upper outline estimates the normal faint-edge thickness;
only an unusually thick lower fringe is trimmed. A detached drop retains every
main-threshold pixel. At initial contact, a nearly complete ellipse with at most
5% truncation below the surface can constrain diffuse outer fringes. A normalized
gradient check preserves sharp thin-sheet edges; deformed silhouettes receive no
shape correction. The correction runs after outline cleanup so cleanup cannot
reconnect the rejected shadow, and it never replaces the whole silhouette with
an ellipse. CLI, preview seeking, playback, and threshold-contact sampling use
the same preceding filename frames, independent of viewing order.

This conservative guard leaves ambiguous cases unchanged. In particular, it
cannot reliably correct a main-threshold silhouette distorted by shadows or
resolve a gap smaller than its edge uncertainty. The preview and CSV flag
possible shadow contact when only the faint candidate reaches the surface,
and contact onset immediately following a tracked detached drop.
Check the raw lower contour and calibrated surface before trusting such contact.
Disable **Approach / onset shadow guard** or use `--no-approach-shadow-guard` to
compare with ordinary faint-region growth.

The guard targets approach and initial contact, not general contact detection
during recoil or lift-off. A faint neck during those later stages can still
produce a nonzero contact width and a review warning. Reducing weak growth to
remove such a neck can also remove real faint liquid near maximum spreading;
inspect both stages before changing the growth limit.

`--weak-growth-steps 0` disables faint extension. Positive values bound the number
of pixel-neighbor growth steps; increasing this limit permits more growth,
whereas increasing the weak change threshold rejects weaker changes.

The automatic strong change threshold has a floor (`--min-strong-change-threshold`). A first
noise estimate locates a sampling anchor: first contact with a calibrated line,
or sustained loss of ellipse agreement when the surface is uncalibrated. Shape
detection requires three complete elliptical silhouettes and two consecutive
deformed silhouettes of the tracked drop; it only chooses threshold samples,
and does not set the reported impact frame. The sample window contains two
frames before the anchor, the anchor itself, and seven frames after it (at most
10 frames), clipped to the working range. With no anchor, up to eight evenly
spaced frames cover the range rather than just its beginning.

For each sample, background differences more than 30 px above the tilted line
give a robust noise level (median + 8 MAD-based standard deviations). The 90th
percentile of those per-frame levels, bounded by the floor, sets the threshold.
Explicit thresholds bypass sampling.

For missing liquid edges, lower the strong or weak change thresholds, or increase the weak
growth limit. For separate specks, increase minimum drop area or outline cleanup size.

### Substrate calibration

One substrate line is used for the whole sequence. The surface calibration
frame and `--surface-calibration-method` determine how that line is found:

| Reflection mode | Surface calibration |
| --- | --- |
| `mirror` | Uses the drop and its reflection. |
| `none` | Uses the drop's lower edge; suitable for weak reflections. |
| `auto` | Selects `mirror` only with strong reflection evidence; assumes `mirror` when no surface calibration frame is supplied. |

`--surface-y-px` overrides the line's center position. Surface angles are
clockwise; without calibration or an explicit angle, the fallback is `0` deg.

Use **Select surface line manually** on an impacted frame when the automatic
candidate is wrong. A manually supplied position and tilt bypass silhouette line
fitting, even if a calibration frame is also selected.

### Spreading diameter, impact, and time

`d_spread` is the outermost liquid span within the substrate band
(`--contact-band-half-height-px`, default 1 px); it is zero when the drop is clear of that
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

Each run writes measurement results, analysis settings, and diagnostic images:

| File / directory | Contents |
| --- | --- |
| `cv_output.csv` | Per-frame measurements and reference droplet dimensions. |
| `run_params.json` | Runtime settings and package version. |
| `debug_overlays/` | Frame images with detected outlines and measured diameters. |

### Output locations

Use `--output-dir` (or `-o`) to select the output folder.

| Mode / GUI output choice | Output folder |
| --- | --- |
| CLI default / GUI: Project outputs | `<project-dir>/outputs/<input-folder-name>/` |
| GUI: Input folder (default) | `<input-folder>/CV_postprocessing/` |
| GUI: Custom folder | Selected directory |

`<project-dir>` contains `pyproject.toml`. The GUI selects the location configured
in `cv_config.txt`, defaulting to the input-folder choice when no output paths
are configured.

`--output-csv` and `--debug-dir` override the CSV and overlay paths individually.
Without `--output-dir`, the CSV path's parent determines the output folder,
followed by the debug folder's parent if only `--debug-dir` is set.
`run_params.json` is saved in the output folder.

### Diagnostic overlays

Overlays are written every frame by default; `--debug-interval` controls the
regular interval. The first frame, size reference frame, and frames near impact
can also produce overlays. Each full run first removes existing frame-numbered
overlay PNGs from the target debug directory.

### CSV columns and naming

Column names use lowercase snake_case. `d` means diameter, `eq` means equivalent,
and `ref_` marks quantities frozen at the size reference frame. Suffixes indicate
units: `_px`, `_mm`, `_ms`, and `_px2` for area.

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
| `shadow_removed_px2` | Candidate pixels rejected by the near-surface shadow guard before final measurement; may include below-surface reflection pixels. |
| `segmentation_warning` | Nonempty when faint-only surface contact needs visual review; an empty value does not certify an accurate boundary. |
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

### Names and units

GUI threshold labels refer to **background change**, not raw image brightness.
Both brightening and darkening count. **Weak growth limit (steps)** is a count
of 4-neighbor propagation steps through eligible pixels before cleanup, not a
Euclidean radius or a strict bound on the final cleaned contour. Use
`--weak-growth-steps` to set this limit.

**Min gas-side area (px2)** is the total detected area on or above the surface,
including detached drops; it is not contact-line length. Set it with
`--min-gas-area-px2`. **Min component area**
instead filters the whole candidate component (which may include reflections)
and is also applied when retaining a component after surface clipping.

**Outline cleanup radius (px)** defines an elliptical kernel of width
`2*radius+1`. Radius 0 disables closing/opening morphology; enclosed-hole
filling and component selection still apply.

**Contact-band half-height (px)** is vertical y tolerance above and below the
line at each x, not full band thickness or surface-normal distance. **Surface
calibration method** selects how the fixed line is calibrated: `none` uses the
lower contour without mirror symmetry. It does not disable surface calibration
or switch shadow suppression on or off.

### Substrate and width measurement

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--droplet-size-ref-frame` | automatic | Frame used for both reference equivalent diameters; Auto uses the first frame whose substrate clearance is at most half its volume-equivalent diameter. |
| `--rim-to-rim {0,1}` | `0` | Set to `1` to measure the silhouette's rim-to-rim width instead of its contact-line width. |
| `--substrate` | `None` | Surface name; experiment metadata only. |
| `--hydrophilicity {SHI,HI,HO,SHO}` | `None` | Surface wettability; experiment metadata only. |
| `--substrate_temp_deg_c` | `None` | Surface temperature in degrees Celsius; experiment metadata only. |
| `--mirror-min-depth-px` | `50` | Minimum mirror candidate depth below the coarse surface line; also used for automatic reflection classification. |
| `--surface-calibration-method {auto,mirror,none}` | `auto` | Select automatic, reflection-symmetry, or lower-contour surface calibration. |
| `--surface-angle-deg` | detected or `0` | Manually override the clockwise surface angle. |
| `--contact-band-half-height-px` | `1` | Vertical half-height of the contact-measurement band at each x, not full thickness or normal distance. |
| `--surface-calibration-frame` | none | Impacted-droplet frame used to calibrate the fixed surface line. |
| `--surface-y-px` | automatic | Manually override the surface-line center y coordinate. |

### Liquid detection

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--foreground-depth-px` | `220` | Candidate-search depth vertically below the line; reported area remains clipped to the gas side. |
| `--approach-shadow-guard`, `--no-approach-shadow-guard` | enabled | Use contour, motion, and edge evidence to limit lower shadow fringes before and at initial contact. |
| `--min-strong-change-threshold` | `200` | Lower bound used by automatic strong-change-threshold estimation. |
| `--strong-change-threshold` | automatic | Strong threshold on absolute background change in gray levels, not raw image brightness. |
| `--top-exclusion-ratio` | `0.25` | Fraction of image height excluded from droplet detection at the top, between 0 and 1. Background and surface calibration still use the full image. |
| `--weak-growth-steps` | `40` | Maximum 4-neighbor weak-growth steps before cleanup; not a Euclidean radius. 0 disables weak extension. |
| `--weak-change-threshold` | ratio-derived | Explicit weak threshold for faint liquid. |
| `--weak-change-threshold-ratio` | `0.3` | Weak/strong threshold ratio when no explicit weak threshold is set. |

### Drop selection and outline cleanup

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--min-gas-area-px2` | `30` | Minimum total detected gas-side area, not contact width; also applies to detached drops. |
| `--min-component-area-px2` | `250` | Minimum candidate-component area, also used after surface clipping. |
| `--morphology-radius-px` | `3` | Elliptical cleanup radius; kernel width is `2*radius+1`. 0 disables gap/speck morphology; hole filling remains. |

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
| `--output-dir`, `-o` | `<project-dir>/outputs/<input-folder-name>` | Folder containing `cv_output.csv`, `run_params.json`, and `debug_overlays/`. |
| `--output-csv` | `<output-dir>/cv_output.csv` | Override the CSV path. |
| `--debug-dir` | `<output-dir>/debug_overlays` | Override the diagnostic folder. |
| `--debug-interval` | `1` | Write one debug overlay every N frames. |

## Development

Run the test suite with:

```bash
uv run python -m unittest discover -v
```
