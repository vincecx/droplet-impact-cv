# droplet-impact-cv

Measure droplet spreading diameter from side-view high-speed TIFF/JPEG sequences,
using a fully automatic CLI or a semi-automatic GUI.

Input images must be grayscale TIFF or 8-bit JPEG, with a six-digit frame number
at the end of each filename. For example, `capture_000005.tif` is reported as
frame 5 even when it is the first file in the input directory. The first frame in
the selected working range must be a clean background with no droplet.

## Usage

```bash
uv sync
```

### Fully automatic CLI mode

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

Select one input folder in the GUI; command-line analysis overrides are not
accepted. Settings follow **GUI changes > cv_config.txt > program defaults**.
Changes are not written back to `cv_config.txt`. Reset restores the values loaded
when the folder was opened, including their Auto state.

The initial preview is `ref_frame`, or the first frame if no reference is found.
Preview changes process the selected frame and any required calibration/threshold
updates, without rerunning the full sequence or writing outputs. Full processing
uses the current GUI settings for the configured working range.

Playback speed (default 5 fps) is separate from camera `fps`. Playback waits for
each frame to finish processing if the requested speed cannot be reached.
The GUI requires Python with Tk support (`tkinter`).

### Processing rules

Liquid is detected by changes from the clean background, whether darker or
brighter. The main threshold identifies clear liquid; the faint threshold extends
the outline only near already detected liquid. Enclosed bright regions are filled
as part of the drop.

One substrate line is used for the whole sequence. `--reflection-mode mirror`
uses the drop and its reflection; `none` uses the drop's lower edge. `auto` selects
`mirror` only with strong reflection evidence, or assumes it when no calibration
frame is supplied. Use `none` for weak reflections. Surface angles are clockwise;
without calibration or an explicit angle, the fallback is `-0.6` deg.
`--surface-y-px` overrides the line's center position.

### CLI parameter summary

Put experiment settings in `<input-folder>/cv_config.txt`; blank lines and `#`
comments are allowed. CLI precedence is **explicit options > cv_config.txt >
program defaults**. Example:

```text
--fps 4000
--pixel-size-mm-per-px 0.01682736321
--calibration-frame 9105
--reflection-mode mirror
```

#### Key parameters

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
| `--strong-threshold` | automatic | Main liquid-detection threshold; lower values include fainter liquid and more noise. |
| `--min-strong-threshold` | `1500` | Lower bound used by automatic strong-threshold estimation. |
| `--weak-threshold` | ratio-derived | Explicit weak threshold for faint liquid. |
| `--weak-threshold-ratio` | `0.25` | Weak/strong threshold ratio when no explicit weak threshold is set. |
| `--weak-reach-px` | `40` | Maximum distance faint liquid can extend from clearly detected liquid. |
| `--min-droplet-area-px2` | `250` | Ignore detected objects smaller than this area. |
| `--morphology-radius-px` | `3` | Cleanup size for closing gaps and removing small specks. |
| `--surface-band-px` | `3` | Half-height of the band in which spreading width is measured. |
| `--hydrophobic {0,1}` | `0` | Set to `1` for hydrophobic surfaces to measure the silhouette's rim-to-rim width instead of its contact-line width. |

For missing liquid edges, lower the main/faint thresholds or increase faint
reach. For separate specks, increase minimum drop area or outline cleanup size.

#### Other parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `-o`, `--output-csv` | `<project-dir>/outputs/<input-folder-name>/spreading_diameter.csv` | CSV output path. |
| `--foreground-depth-px` | `220` | Search depth below the substrate line. |
| `--min-contact-px` | `30` | Minimum detected pixels above the substrate for a selected drop. |
| `--min-impact-width-px` | `30` | Contact-line diameter that marks the first impact frame. |
| `--include-pre-impact`, `--no-include-pre-impact` | disabled | Include or exclude rows before impact. |
| `--time-zero {impact,first-frame}` | `impact` | Choose the time origin. |
| `--debug-dir` | `<project-dir>/outputs/<input-folder-name>/debug_overlays` | Diagnostic overlay directory. |
| `--debug-interval` | `1` | Write one debug overlay every N frames. |

## Outputs

The default CSV filename is `spreading_diameter.csv`.

| Mode / GUI output choice | Output folder |
| --- | --- |
| Automatic / Project outputs | `<project-dir>/outputs/<input-folder-name>/` |
| GUI: Input folder (default) | `<input-folder>/CV_postprocessing/` |
| GUI: Custom folder | Selected directory |

`<project-dir>` is the directory containing `pyproject.toml`; the automatic
default does not depend on the input location or current working directory.
CLI paths can be overridden with `--output-csv` and `--debug-dir`.

GUI output selection overrides configured output paths, preserves the configured
CSV filename, and always puts overlays in `<output-folder>/debug_overlays/`.
For example, the input-folder choice for `sourcedata/example` writes to
`sourcedata/example/CV_postprocessing/`.

`d_spread` is the outermost liquid span within the substrate band
(`--surface-band-px`, default 3 px); it is zero when the drop is clear of that
band. With `--hydrophobic 1`, it is the full rim-to-rim width instead. Impact is
always detected from contact-line width reaching `--min-impact-width-px`.

By default, time is zero at impact and pre-impact rows are omitted.
`--include-pre-impact` retains them; `--time-zero first-frame` changes the origin.

### Reference equivalent diameters

`D_parallel` and `D_normal` are the full silhouette spans parallel and normal
to the substrate. `ref_frame` is the first frame where the lower-contour distance
to the substrate is at most `D_eq_volume / 2`.

Both equivalent diameters are measured in that same reference frame:

- `ref_d_eq_volume`: `(D_parallel² × D_normal)^(1/3)`, the equivalent sphere diameter of an axisymmetric ellipsoid with the measured spans.
- `ref_d_eq_area`: `sqrt(4 × A / pi)`, the equivalent circle diameter of the measured side-view silhouette area `A`.

Both methods use the filled silhouette before surface clipping. The area method
measures a 2D projection, not volume. Reference dimensions and clearance are
frozen at `ref_frame` and repeated on every CSV row; if no reference is found,
those columns are empty.

Overlays are written every frame by default (`--debug-interval` controls this).
Each full run first removes existing frame-numbered overlay PNGs from the target
debug directory.

Parameter names use lowercase snake_case in Python and CSV, and corresponding
hyphenated names in the CLI. `d` means diameter and `eq` means equivalent;
`ref_` marks quantities frozen at the reference frame. Unit suffixes are
`_px`, `_mm`, `_ms`, `_deg`, and `_px2` for area. Pixel size uses `_mm_per_px`;
`fps` is the camera frame rate.

### CSV columns

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

## Development

```bash
uv run python -m unittest discover -v
```
