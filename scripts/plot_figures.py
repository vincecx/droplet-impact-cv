#!/usr/bin/env python3
"""
plot_figures.py — Publication-quality result figures for droplet-impact
spreading measurements.

Standalone script (independent of the droplet_impact_cv analysis workflow).
It reads one or more ``cv_output.csv`` result files produced by the
image-analysis pipeline and renders Nature-style multi-panel figures organized
by physical meaning rather than one plot per variable.

Physics organization
---------------------
Single run  -> ``spreading_dynamics_<tag>``  (asymmetric 4-panel grid)
    a  Spreading factor beta(t) = d/D0          — primary dynamics (hero)
       with contact-line / rim diameters when they differ, beta_max,
       phase shading (spreading vs recoil), optional dimensionless top
       axis tau = tU/D0
    b  Incoming droplet geometry (D_parallel, D_normal, D_eq,V, D_eq,A)
       — setup / initial anisotropy that defines the length scale
    c  Contact-line velocity u_cl(t)            — kinematic mechanism
    d  Lamella morphology: side-profile (silhouette) area A_s/A_s,0 and
       mean thickness h/D0 = A_s/(d D0)   — orthogonal shape evidence
       (the camera is side-looking, so the area column is the droplet
       cross-sectional silhouette, not a top-view footprint)

Multiple runs -> ``spreading_comparison``
    a  beta(t) overlay with direct labels       — cross-condition dynamics
    b  beta_max bar summary
    c  t_max bar summary
Runs are ordered and colored by substrate temperature (cool -> warm) and
tagged by their condition (e.g. "52.3 C #1") when run_params.json carries
the experimental conditions; explicit --labels override this behaviour.

A machine-readable ``spreading_metrics_summary.csv`` is always written.

Time window
-----------
All time-dependent panels show only the dynamics window 0 <= t <= 2*t_max
(see ``dynamics_window``); later frames are not used in the figures. Scalar
metrics and the summary table still refer to the complete CSV.

Inputs
------
Measurements are taken from the current ``cv_output.csv``: time
(time_ms), frame rate (fps), image scale, diameters, impact frame and
component area are read directly from the CSV (or computed from CSV columns);
the script never reads cv_config.txt from the input folder. A
``run_params.json`` in the same result folder supplies settings that are not
stored in the CSV (e.g. the configured surface angle) and is used when
present.

An output folder from ``--output-dir`` can be supplied directly; it resolves to
``cv_output.csv`` in that folder. Only the current CSV columns and run parameter
format are accepted. Explicit custom CSV paths remain usable.

Adding figures
--------------
Each panel is an independent ``build_panel_*`` function that draws onto a
given Axes; the ``figure_*`` functions only arrange them. New panels or new
data columns should be added to the RunData / RunMetrics dataclasses and
wired into a panel builder — no figure-assembly changes are required.

Examples
--------
  uv run python scripts/plot_figures.py outputs/example_tilted_droplet
  uv run python scripts/plot_figures.py outputs/example_tilted_droplet/cv_output.csv
  uv run python scripts/plot_figures.py outputs/run_a/cv_output.csv \
      outputs/run_b/cv_output.csv --labels "We=50" "We=150" \
      --impact-velocity 1.42 --density 998 --surface-tension 0.072 \
      --viscosity 0.001 --tilt 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ===========================================================================
# 1. Data model
# ===========================================================================


@dataclass
class RunData:
    """Raw time series and metadata parsed from one result folder."""

    tag: str
    csv_path: Path
    frame: np.ndarray
    t_ms: np.ndarray
    d_spread: np.ndarray
    d_contact: np.ndarray
    d_rim: np.ndarray
    d_eq_v: np.ndarray
    d_eq_a: np.ndarray
    d_parallel: np.ndarray
    d_normal: np.ndarray
    clearance: np.ndarray
    area_px2: np.ndarray
    fps: float | None
    pixel_mm_per_px: float | None
    impact_frame: float | None
    surface_angle_deg: float | None
    # Experimental conditions recorded in run parameters.
    substrate: str | None = None
    hydrophilicity: str | None = None
    substrate_temp_deg_c: float | None = None

    @property
    def n_rows(self) -> int:
        return int(len(self.t_ms))


@dataclass
class RunMetrics:
    """Derived quantities for one run; arrays align with RunData rows."""

    tag: str
    d0_mm: float
    d0_source: str
    beta: np.ndarray
    beta_max: float
    t_max_ms: float
    idx_max: int
    beta_final: float
    u_cl: np.ndarray
    u_peak_m_s: float
    t_u_peak_ms: float
    aspect_ratio: float | None
    we: float | None
    re: float | None


# ===========================================================================
# 2. Loading (CSV is the data source; run_params.json is optional)
# ===========================================================================

RUN_PARAMS_FILENAME = "run_params.json"
OUTPUT_CSV_FILENAME = "cv_output.csv"
REQUIRED_COLUMNS = (
    "filename", "frame_number", "time_ms", "d_spread_px", "d_spread_mm",
    "d_contact_px", "d_contact_mm", "d_rim_px", "d_rim_mm",
    "droplet_size_ref_frame", "ref_d_eq_volume_px", "ref_d_eq_volume_mm",
    "ref_d_eq_area_px", "ref_d_eq_area_mm", "ref_d_parallel_px", "ref_d_parallel_mm",
    "ref_d_normal_px", "ref_d_normal_mm", "ref_clearance_px", "ref_clearance_mm",
    "impact_frame", "surface_y_px", "droplet_area_px2", "fps",
    "pixel_size_mm_per_px", "surface_calibration_frame",
)


def result_csv_path(path: Path) -> Path:
    """Accept a current output folder or an explicit CSV path override."""
    if path.is_dir():
        return path / OUTPUT_CSV_FILENAME
    if path.name == "spreading_diameter.csv":
        raise ValueError("spreading_diameter.csv is no longer supported; regenerate cv_output.csv.")
    return path


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in ("nan", "none"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_run(path: Path) -> RunData:
    """Parse a current-format spreading CSV; run_params.json is optional."""
    path = result_csv_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            raise ValueError(
                f"{path} is not a current-format spreading CSV "
                f"(missing column(s): {', '.join(missing)}). Regenerate it "
                "with the latest image-analysis pipeline."
            )
        rows = list(reader)

    def column(name: str) -> np.ndarray:
        values = [_to_float(r.get(name)) for r in rows]
        return np.array(
            [np.nan if v is None else v for v in values], dtype=float
        )

    def scalar(name: str) -> float | None:
        arr = column(name)
        finite = arr[np.isfinite(arr)]
        # reference quantities are frozen and identical on every row
        return float(finite[0]) if finite.size else None

    # Settings absent from the CSV (e.g. surface angle) come from the optional
    # run_params.json; all measurements are taken from the CSV itself.
    params: dict = {}
    params_path = path.parent / RUN_PARAMS_FILENAME
    if params_path.is_file():
        try:
            payload = json.loads(params_path.read_text(encoding="utf-8"))
            params = payload.get("parameters", {}) if isinstance(payload, dict) else {}
            if not isinstance(params, dict) or any(
                name not in params for name in ("output_dir", "output_csv", "debug_dir")
            ):
                raise ValueError(f"{params_path} is not a current-format run parameters file; regenerate the results.")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {params_path}: {error}") from None

    data = {
        "tag": path.parent.name,
        "csv_path": path,
        "frame": column("frame_number"),
        "t_ms": column("time_ms"),
        "d_spread": column("d_spread_mm"),
        "d_contact": column("d_contact_mm"),
        "d_rim": column("d_rim_mm"),
        "d_eq_v": column("ref_d_eq_volume_mm"),
        "d_eq_a": column("ref_d_eq_area_mm"),
        "d_parallel": column("ref_d_parallel_mm"),
        "d_normal": column("ref_d_normal_mm"),
        "clearance": column("ref_clearance_mm"),
        "area_px2": column("droplet_area_px2"),
        "fps": scalar("fps"),
        "pixel_mm_per_px": scalar("pixel_size_mm_per_px"),
        "impact_frame": scalar("impact_frame"),
        "surface_angle_deg": _to_float(
            str(params.get("surface_angle_deg", ""))
        ),
        "substrate": (params.get("substrate") or None),
        "hydrophilicity": (params.get("hydrophilicity") or None),
        "substrate_temp_deg_c": _to_float(
            str(params.get("substrate_temp_deg_c", ""))
        ),
    }

    # keep only rows with a valid time axis, sort chronologically
    valid = np.isfinite(data["t_ms"])
    n = len(rows)
    if valid.sum() >= 2:
        order = np.argsort(data["t_ms"][valid], kind="stable")
        for key, val in list(data.items()):
            if isinstance(val, np.ndarray) and val.shape == (n,):
                data[key] = val[valid][order]

    return RunData(
        tag=data["tag"],
        csv_path=data["csv_path"],
        frame=data["frame"],
        t_ms=data["t_ms"],
        d_spread=data["d_spread"],
        d_contact=data["d_contact"],
        d_rim=data["d_rim"],
        d_eq_v=data["d_eq_v"],
        d_eq_a=data["d_eq_a"],
        d_parallel=data["d_parallel"],
        d_normal=data["d_normal"],
        clearance=data["clearance"],
        area_px2=data["area_px2"],
        fps=data["fps"],
        pixel_mm_per_px=data["pixel_mm_per_px"],
        impact_frame=data["impact_frame"],
        surface_angle_deg=data["surface_angle_deg"],
        substrate=data["substrate"],
        hydrophilicity=data["hydrophilicity"],
        substrate_temp_deg_c=data["substrate_temp_deg_c"],
    )


# ===========================================================================
# 3. Physics / derived quantities
# ===========================================================================


def reference_diameter(run: RunData, source: str) -> tuple[float, str]:
    """Return (D0 in mm, provenance). source: auto | volume | area."""
    v_arr, a_arr = run.d_eq_v, run.d_eq_a
    v = v_arr[np.isfinite(v_arr)][0] if np.isfinite(v_arr).any() else np.nan
    a = a_arr[np.isfinite(a_arr)][0] if np.isfinite(a_arr).any() else np.nan

    if source == "volume" and math.isfinite(v):
        return float(v), "volume-equivalent"
    if source == "area" and math.isfinite(a):
        return float(a), "area-equivalent"
    if source == "auto":
        if math.isfinite(v):
            return float(v), "volume-equivalent"
        if math.isfinite(a):
            return float(a), "area-equivalent"
    # last resort: first measured spread diameter
    d0 = run.d_spread[0] if run.d_spread.size else np.nan
    if math.isfinite(d0):
        return float(d0), "first-frame"
    raise ValueError(
        f"No reference diameter could be determined from {run.csv_path}."
    )


def smooth_derivative(t_s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dy/dt via Savitzky-Golay (noise robust) with np.gradient fallback."""
    mask = np.isfinite(t_s) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan)
    dt = float(np.median(np.diff(t_s[mask])))
    if dt <= 0:
        dt = 1.0
    try:
        from scipy.signal import savgol_filter

        count = mask.sum()
        window = min(7, count if count % 2 == 1 else count - 1)
        while window > 5 and window % 2 == 0:
            window -= 1
        polyorder = 2 if window >= 5 else 1
        out = np.full_like(y, np.nan)
        out[mask] = savgol_filter(
            y[mask], window_length=window, polyorder=polyorder,
            deriv=1, delta=dt, mode="interp",
        )
        return out
    except Exception:
        return np.gradient(y, t_s, edge_order=1)


def aspect_ratio(run: RunData) -> float | None:
    p, n = run.d_parallel, run.d_normal
    if np.isfinite(p).any() and np.isfinite(n).any() and n[0] != 0:
        return float(p[0] / n[0])
    return None


# Only the dynamics window [0, 2*t_max] is shown in the figures.
WINDOW_FACTOR = 2.0


def dynamics_window(run: RunData, m: RunMetrics,
                    factor: float = WINDOW_FACTOR) -> np.ndarray:
    """Boolean mask of frames belonging to the plotted dynamics window."""
    t = run.t_ms
    if not math.isfinite(m.t_max_ms) or m.t_max_ms <= 0:
        return np.isfinite(t)
    return (
        np.isfinite(t)
        & (t >= 0.0)
        & (t <= factor * m.t_max_ms + 1e-9)
    )


def evaluate_run(run: RunData, *, d0_mm: float, d0_source: str,
                 impact_velocity: float | None = None,
                 rho: float | None = None, sigma: float | None = None,
                 mu: float | None = None) -> RunMetrics:
    """Compute the derived quantities of one run."""
    t, d = run.t_ms, run.d_spread
    beta = d / d0_mm if math.isfinite(d0_mm) else d.copy()

    finite = np.isfinite(beta)
    idx_max = int(np.nanargmax(beta)) if finite.any() else -1
    beta_max = float(beta[idx_max]) if idx_max >= 0 else math.nan
    t_max = float(t[idx_max]) if idx_max >= 0 else math.nan
    beta_final = float(beta[finite][-1]) if finite.any() else math.nan

    u_cl = smooth_derivative(t / 1000.0, d / 1000.0)  # m/s
    positive = u_cl > 0
    u_peak = float(np.nanmax(u_cl)) if positive.any() else math.nan
    if positive.any():
        t_u_peak = float(
            t[int(np.nanargmax(np.where(positive, u_cl, np.nan)))]
        )
    else:
        t_u_peak = math.nan

    we = reynolds = None
    if impact_velocity is not None and math.isfinite(d0_mm):
        d0_m = d0_mm / 1000.0
        if rho and sigma:
            we = rho * impact_velocity**2 * d0_m / sigma
        if rho and mu:
            reynolds = rho * impact_velocity * d0_m / mu

    return RunMetrics(
        tag=run.tag,
        d0_mm=d0_mm,
        d0_source=d0_source,
        beta=beta,
        beta_max=beta_max,
        t_max_ms=t_max,
        idx_max=idx_max,
        beta_final=beta_final,
        u_cl=u_cl,
        u_peak_m_s=u_peak,
        t_u_peak_ms=t_u_peak,
        aspect_ratio=aspect_ratio(run),
        we=we,
        re=reynolds,
    )


# ===========================================================================
# 4. Presentation constants and shared helpers
# ===========================================================================

PALETTE = {
    "hero": "#0F4D92",
    "contact": "#42949E",
    "rim": "#4D4D4D",
    "neutral_mid": "#767676",
    "up": "#0F4D92",
    "down": "#B64342",
    "morph": "#3775BA",
    "thickness": "#9A4D8E",
}

# restrained categorical sequence for multi-run overlays
RUN_COLORS = [
    "#0F4D92", "#42949E", "#9A4D8E", "#B64342",
    "#3775BA", "#767676", "#5B8FD6",
]

# small white background for annotations drawn over curves/grid lines
ANNOTATION_BOX = dict(boxstyle="round,pad=0.18", fc="white",
                      ec="none", alpha=0.9)


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",   # editable text in SVG
        "pdf.fonttype": 42,       # editable TrueType in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def panel_label(ax, label: str) -> None:
    """Bold lowercase panel label with a fixed point offset from the corner."""
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(
        -6 / 72, 5 / 72, ax.figure.dpi_scale_trans
    )
    ax.text(
        0.0, 1.0, label,
        transform=ax.transAxes + offset,
        fontsize=9.5, fontweight="bold", ha="left", va="bottom",
    )


def save_figure(fig, outdir: Path, stem: str, formats: list[str],
                dpi: int) -> list[Path]:
    paths: list[Path] = []
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = outdir / f"{stem}.{fmt}"
        fig.savefig(
            target,
            dpi=dpi if fmt in ("png", "tiff", "tif", "jpg") else None,
            bbox_inches="tight", pad_inches=0.05,
        )
        paths.append(target)
    return paths


def _series_differs(a: np.ndarray, b: np.ndarray) -> bool:
    """True where b differs from a by more than 1% of a's scale."""
    if a.shape != b.shape:
        return False
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return False
    scale = np.nanmax(np.abs(a[mask])) or 1.0
    return float(np.nanmax(np.abs(a[mask] - b[mask])) / scale) > 0.01


def add_phase_labels(ax, m: RunMetrics, win_end: float) -> None:
    """Spreading / Recoil labels fixed to the inner top of the axes."""
    trans = ax.get_xaxis_transform()  # x = data, y = axes fraction
    kw = dict(transform=trans, ha="center", va="top",
              fontsize=6.6, color=PALETTE["neutral_mid"])
    ax.text(m.t_max_ms / 2.0, 0.965, "Spreading", **kw)
    if win_end > m.t_max_ms * 1.15:
        ax.text((m.t_max_ms + win_end) / 2.0, 0.965,
                "Recoil / relaxation", **kw)


def condition_summary(run: RunData, m: RunMetrics, u0: float | None,
                      tilt: float | None) -> str:
    """One-line condition metadata rendered above the figure, not inside axes."""
    parts = [rf"$D_0$ = {m.d0_mm:.2f} mm"]
    if u0 is not None:
        parts.append(rf"$U_0$ = {u0:.2f} m s$^{{-1}}$")
    if m.we is not None:
        parts.append(rf"$We$ = {m.we:.0f}")
    if m.re is not None:
        parts.append(rf"$Re$ = {m.re:.0f}")
    if tilt is not None:
        parts.append(rf"$\alpha$ = {tilt:.2f}$\degree$")
    if run.substrate_temp_deg_c is not None:
        parts.append(rf"$T_s$ = {run.substrate_temp_deg_c:g}$\degree$C")
    return "     ".join(parts)


def effective_tilt(run: RunData, cli_tilt: float | None) -> float | None:
    """CLI --tilt wins; otherwise the angle recorded in run_params.json."""
    return cli_tilt if cli_tilt is not None else run.surface_angle_deg


# ===========================================================================
# 5. Single-run panel builders
# ===========================================================================


def build_panel_beta(ax, run: RunData, m: RunMetrics, *,
                     impact_velocity: float | None = None,
                     window: np.ndarray | None = None) -> None:
    """Panel a: spreading factor beta(t), primary dynamics."""
    w = np.ones(run.n_rows, dtype=bool) if window is None else window
    t = run.t_ms[w]
    win_end = float(t[-1])
    spread = run.d_spread[w]

    # phase shading and labels (labels sit clear of the curve via headroom)
    ax.axvspan(0.0, m.t_max_ms, color=PALETTE["hero"], alpha=0.06, lw=0)
    add_phase_labels(ax, m, win_end)

    # alternative diameter definitions, only when meaningfully different
    d_contact, d_rim = run.d_contact[w], run.d_rim[w]
    if _series_differs(spread, d_contact):
        ax.plot(t, d_contact / m.d0_mm, color=PALETTE["contact"],
                lw=1.3, ls=(0, (4, 2)), label="Contact line")
    if _series_differs(spread, d_rim):
        ax.plot(t, d_rim / m.d0_mm, color=PALETTE["rim"],
                lw=1.3, ls=(0, (1.5, 1.5)), label="Rim to rim")

    ax.plot(t, m.beta[w], color=PALETTE["hero"], lw=1.8, marker="o",
            ms=3.4, mfc="white", mew=1.1, label=r"Spreading, $\beta$")

    ax.axhline(1.0, color=PALETTE["neutral_mid"], lw=0.7, ls=(0, (3, 3)))
    # boundary line stops below the peak annotation
    ax.axvline(m.t_max_ms, color=PALETTE["neutral_mid"], lw=0.6,
               ls=(0, (2, 3)), ymax=0.84)

    ax.plot(m.t_max_ms, m.beta_max, marker="v", ms=5,
            color=PALETTE["hero"], mfc=PALETTE["hero"], zorder=5)
    ax.annotate(
        rf"$\beta_{{\max}}={m.beta_max:.2f}$, $t={m.t_max_ms:.2f}$ ms",
        xy=(m.t_max_ms, m.beta_max),
        xytext=(0, 5), textcoords="offset points",
        fontsize=6.6, ha="center", va="bottom", bbox=ANNOTATION_BOX,
    )

    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Spreading factor $\beta = d/D_0$")
    ax.set_xlim(0.0, win_end * 1.03)
    ax.set_ylim(top=m.beta_max * 1.24)
    ax.locator_params(axis="y", nbins=6)

    # dimensionless top axis when impact velocity is supplied:
    # tau = (t_ms/1000) U / (D_mm/1000) = t_ms U / D_mm
    if impact_velocity is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xbound(ax.get_xbound())
        ticks = ax.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(
            [f"{tk * impact_velocity / m.d0_mm:.1f}" for tk in ticks]
        )
        ax2.set_xlabel(r"Dimensionless time $\tau = tU_0/D_0$",
                       fontsize=7)
        ax2.spines["top"].set_visible(True)
        ax2.spines["top"].set_linewidth(0.8)
        ax2.tick_params(length=2.0, width=0.7, labelsize=6.5)

    ax.legend(loc="lower right", ncol=1, handlelength=1.8)


def build_panel_geometry(ax, run: RunData, m: RunMetrics) -> None:
    """Panel b: incoming droplet reference geometry (horizontal lollipop)."""
    entries: list[tuple[str, float]] = []
    if np.isfinite(run.d_parallel).any():
        entries.append((r"$D_\parallel$", float(run.d_parallel[0])))
    if np.isfinite(run.d_eq_a).any():
        entries.append((r"$D_{\mathrm{eq,A}}$", float(run.d_eq_a[0])))
    if np.isfinite(run.d_eq_v).any():
        entries.append((r"$D_{\mathrm{eq,V}}$", float(run.d_eq_v[0])))
    if np.isfinite(run.d_normal).any():
        entries.append((r"$D_\perp$", float(run.d_normal[0])))

    if not entries:
        ax.text(0.5, 0.5, "Reference geometry\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=PALETTE["neutral_mid"])
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ys = np.arange(len(entries))[::-1]
    vals = np.array([e[1] for e in entries])
    x_left = float(vals.min() - (vals.max() - vals.min() + 0.3) * 0.6)
    for y, (_, val) in zip(ys, entries):
        ax.hlines(y, x_left, val, color=PALETTE["neutral_mid"], lw=0.7)
        ax.plot(val, y, marker="o", ms=4, color=PALETTE["hero"])
        ax.text(val + 0.02, y, f"{val:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_xlim(x_left, float(vals.max()) * 1.06)
    ax.set_xlabel("Diameter (mm)")
    title = "Incoming droplet"
    if m.aspect_ratio is not None:
        title += "\n" + rf"$D_\parallel/D_\perp={m.aspect_ratio:.3f}$"
    ax.set_title(title, fontsize=7.3, pad=3)


def build_panel_velocity(ax, run: RunData, m: RunMetrics, *,
                         impact_velocity: float | None = None,
                         window: np.ndarray | None = None) -> None:
    """Panel c: contact-line velocity with sign-coloured regions."""
    w = np.ones(run.n_rows, dtype=bool) if window is None else window
    t = run.t_ms[w]
    win_end = float(t[-1])
    u = m.u_cl[w]

    if impact_velocity is not None:
        u_plot = u / impact_velocity
        ylabel = r"Contact-line velocity $u_{\mathrm{cl}}/U_0$"
    else:
        u_plot = u
        ylabel = r"Contact-line velocity (m s$^{-1}$)"

    ax.plot(t, u_plot, color=PALETTE["hero"], lw=1.4)
    ax.fill_between(t, u_plot, 0, where=(u_plot > 0),
                    color=PALETTE["up"], alpha=0.10, lw=0)
    ax.fill_between(t, u_plot, 0, where=(u_plot < 0),
                    color=PALETTE["down"], alpha=0.10, lw=0)
    ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.7)

    # peak marker + label above the curve (peak lies inside the window)
    if math.isfinite(m.u_peak_m_s) and m.t_u_peak_ms <= win_end:
        upk = (
            m.u_peak_m_s / impact_velocity
            if impact_velocity is not None
            else m.u_peak_m_s
        )
        ax.plot(m.t_u_peak_ms, upk, marker="v", ms=4.5,
                color=PALETTE["hero"], mfc=PALETTE["hero"])
        if impact_velocity is not None:
            text = rf"$u_{{\max}}/U_0={upk:.2f}$"
        else:
            text = rf"$u_{{\max}}={upk:.2f}$"
        ax.annotate(text, xy=(m.t_u_peak_ms, upk),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=6.6, ha="left", va="bottom",
                    bbox=ANNOTATION_BOX)
        ax.set_ylim(top=float(np.nanmax(u_plot)) * 1.32)

    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, win_end * 1.03)


def build_panel_morphology(ax, run: RunData, m: RunMetrics, *,
                           window: np.ndarray | None = None) -> None:
    """Panel d: side-profile silhouette area and mean lamella thickness."""
    w = np.ones(run.n_rows, dtype=bool) if window is None else window
    t = run.t_ms[w]
    win_end = float(t[-1])
    px = run.pixel_mm_per_px
    d_win = run.d_spread[w]

    # normalize by the silhouette area of the incoming spherical droplet
    d0_px = m.d0_mm / px if px else np.nan
    if math.isfinite(d0_px):
        a_s0_px2 = math.pi / 4.0 * d0_px**2
    else:
        a_s0_px2 = float(run.area_px2[w][0])
    a_ratio = run.area_px2[w] / a_s0_px2

    ax.plot(t, a_ratio, color=PALETTE["morph"], lw=1.5,
            label=r"$A_s/A_{s,0}$")
    ax.fill_between(t, a_ratio, float(np.nanmin(a_ratio)),
                    color=PALETTE["morph"], alpha=0.08, lw=0)
    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Side-profile area $A_s/A_{s,0}$",
                  color=PALETTE["morph"])
    ax.tick_params(axis="y", colors=PALETTE["morph"])
    ax.spines["left"].set_color(PALETTE["morph"])
    ax.set_xlim(0.0, win_end * 1.03)

    # frames with a reliable contact width (same mask used for thickness)
    width_ok = d_win > 0.5 * m.d0_mm

    # mean thickness h = silhouette area / spread width; mask early frames
    if px:
        a_mm2 = run.area_px2[w] * px**2
        with np.errstate(divide="ignore", invalid="ignore"):
            h_mm = np.where(width_ok, a_mm2 / d_win, np.nan)
        ax2 = ax.twinx()
        ax2.plot(t, h_mm / m.d0_mm, color=PALETTE["thickness"],
                 lw=1.2, ls=(0, (3, 2)), label=r"$h/D_0$")
        ax2.set_ylabel(r"Mean thickness $h/D_0$",
                       color=PALETTE["thickness"], fontsize=7.5)
        ax2.tick_params(axis="y", colors=PALETTE["thickness"],
                        length=2.6, width=0.7)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(PALETTE["thickness"])
        ax2.spines["right"].set_linewidth(0.8)
        ax2.spines["top"].set_visible(False)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="best",
                  ncol=1, handlelength=1.6, columnspacing=0.8)

    # max flattening = minimum silhouette area among frames that still have
    # a measured contact width, so detection-loss frames cannot be selected
    candidates = np.where(width_ok & np.isfinite(a_ratio))[0]
    if candidates.size == 0:
        candidates = np.where(np.isfinite(a_ratio))[0]
    i_min = int(candidates[np.argmin(a_ratio[candidates])])
    ax.plot(t[i_min], a_ratio[i_min], marker="o", ms=4,
            color=PALETTE["morph"])
    ax.annotate(
        "max flattening\n" + rf"$A_s/A_{{s,0}}={a_ratio[i_min]:.2f}$",
        xy=(t[i_min], a_ratio[i_min]),
        xytext=(8, 12), textcoords="offset points", fontsize=6.6,
        arrowprops=dict(arrowstyle="-", lw=0.6,
                        color=PALETTE["neutral_mid"]),
        bbox=ANNOTATION_BOX,
    )


# ===========================================================================
# 6. Single-run figure assembly
# ===========================================================================


def figure_single(run: RunData, m: RunMetrics, args,
                  outdir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = GridSpec(
        2, 6, figure=fig,
        hspace=0.62, wspace=1.35,
        left=0.07, right=0.985, top=0.86, bottom=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:6])

    window = dynamics_window(run, m)
    build_panel_beta(ax_a, run, m,
                     impact_velocity=args.impact_velocity, window=window)
    build_panel_geometry(ax_b, run, m)
    build_panel_velocity(ax_c, run, m,
                         impact_velocity=args.impact_velocity, window=window)
    build_panel_morphology(ax_d, run, m, window=window)

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, label)

    # condition metadata occupies its own line above the panels
    tilt = effective_tilt(run, args.tilt)
    summary = condition_summary(run, m, args.impact_velocity, tilt)
    fig.text(0.07, 0.955, summary, fontsize=7, ha="left", va="center")

    stem = f"spreading_dynamics_{_safe(run.tag)}"
    return save_figure(fig, outdir, stem, args.formats, args.dpi)


# ===========================================================================
# 7. Comparison panel builders and figure assembly
# ===========================================================================


def derive_tags(runs: list[RunData]) -> list[str]:
    """Short, condition-based display tags.

    Runs carrying a substrate temperature are tagged by temperature
    (e.g. "52.3 °C"); repeated conditions get a "#n" suffix. Runs without
    condition metadata keep their result-folder name.
    """
    bases = [
        f"{r.substrate_temp_deg_c:g} °C"
        if r.substrate_temp_deg_c is not None
        else r.tag
        for r in runs
    ]
    counts = Counter(bases)
    seen: Counter = Counter()
    tags: list[str] = []
    for base in bases:
        seen[base] += 1
        tags.append(f"{base} #{seen[base]}" if counts[base] > 1 else base)
    return tags


def condition_colors(runs: list[RunData]) -> list:
    """Color runs by substrate temperature (cool -> warm).

    Falls back to the restrained categorical sequence when any run lacks
    temperature metadata.
    """
    temps = [r.substrate_temp_deg_c for r in runs]
    if any(t is None for t in temps):
        return [RUN_COLORS[i % len(RUN_COLORS)] for i in range(len(runs))]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "temperature", ["#0F4D92", "#E08214", "#B64342"]
    )
    tmin, tmax = min(temps), max(temps)
    span = max(tmax - tmin, 1e-9)
    return [cmap((t - tmin) / span) for t in temps]


def condition_sort_key(run: RunData):
    """Order comparison runs by temperature, then by result-folder name."""
    temp = run.substrate_temp_deg_c
    return (temp if temp is not None else math.inf,
            run.csv_path.parent.name)


def build_panel_overlay(ax, runs: list[RunData],
                        metrics: list[RunMetrics],
                        colors: list[str]) -> None:
    """Comparison panel a: beta(t) overlay with direct end labels."""
    label_targets: list[tuple[float, float, str, str]] = []
    for run, m, color in zip(runs, metrics, colors):
        w = dynamics_window(run, m)
        t, beta = run.t_ms[w], m.beta[w]
        ax.plot(t, beta, color=color, lw=1.6, label=m.tag)
        ax.plot(m.t_max_ms, m.beta_max, marker="v", ms=4.5,
                color=color, mfc=color)
        label_targets.append((float(t[-1]), float(beta[-1]),
                              m.tag, color))

    ax.axhline(1.0, color=PALETTE["neutral_mid"], lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Spreading factor $\beta = d/D_0$")

    x_end = max(target[0] for target in label_targets)
    ax.set_xlim(0.0, x_end * 1.28)

    # greedy de-collision of direct end labels
    yspan = float(ax.get_ylim()[1] - ax.get_ylim()[0])
    min_dy = 0.075 * yspan
    placed: list[tuple[float, float]] = []
    for x, y, tag, color in sorted(label_targets, key=lambda z: z[1]):
        y_adj = y
        for _, py in placed:
            if abs(y_adj - py) < min_dy:
                y_adj = py + min_dy
        placed.append((x, y_adj))
        ax.plot([x, x_end * 1.04], [y, y_adj], color=color,
                lw=0.5, ls=(0, (2, 2)))
        ax.text(x_end * 1.06, y_adj, tag, color=color,
                fontsize=7, va="center", ha="left")
    ax.set_ylim(top=float(ax.get_ylim()[1]) * 1.06)


def build_panel_summary_bar(ax, tags: list[str], colors: list[str],
                            values: list[float], *, ylabel: str,
                            title: str, refline: float | None = None) -> None:
    """Comparison panel b/c: scalar summary as coloured bars."""
    x = np.arange(len(tags))
    ax.bar(x, values, color=colors, width=0.66, lw=0)
    vmax = max(values)
    for xi, value in zip(x, values):
        ax.text(xi, value + vmax * 0.02, f"{value:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    if refline is not None:
        ax.axhline(refline, color=PALETTE["neutral_mid"],
                   lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=6.3)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_ylim(0.0, vmax * 1.16)
    ax.set_title(title, fontsize=7.5)


def figure_comparison(runs: list[RunData], metrics: list[RunMetrics],
                      args, outdir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = GridSpec(
        2, 6, figure=fig,
        hspace=0.95, wspace=1.3,
        left=0.07, right=0.98, top=0.93, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[:, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 4:6])

    tags = [r.tag for r in runs]
    colors = condition_colors(runs)

    build_panel_overlay(ax_a, runs, metrics, colors)
    build_panel_summary_bar(
        ax_b, tags, colors, [m.beta_max for m in metrics],
        ylabel=r"$\beta_{\max}$", title="Maximum spreading", refline=1.0,
    )
    build_panel_summary_bar(
        ax_c, tags, colors, [m.t_max_ms for m in metrics],
        ylabel=r"$t_{\max}$ (ms)", title="Time to maximum",
    )

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")

    return save_figure(fig, outdir, "spreading_comparison",
                       args.formats, args.dpi)


# ===========================================================================
# 8. Metrics summary CSV
# ===========================================================================


def write_summary(runs: list[RunData], metrics: list[RunMetrics],
                  outdir: Path) -> Path:
    out = outdir / "spreading_metrics_summary.csv"
    columns = [
        "tag", "substrate", "hydrophilicity", "substrate_temp_deg_c",
        "n_frames", "duration_ms", "d0_mm", "d0_source",
        "beta_max", "t_max_ms", "beta_final", "u_peak_m_s",
        "t_u_peak_ms", "aspect_ratio", "weber_number", "reynolds_number",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for run, m in zip(runs, metrics):
            writer.writerow([
                m.tag, run.substrate or "", run.hydrophilicity or "",
                (f"{run.substrate_temp_deg_c:g}"
                 if run.substrate_temp_deg_c is not None else ""),
                run.n_rows,
                f"{float(run.t_ms[-1]):.3f}",
                f"{m.d0_mm:.4f}", m.d0_source,
                f"{m.beta_max:.4f}", f"{m.t_max_ms:.3f}",
                f"{m.beta_final:.4f}",
                f"{m.u_peak_m_s:.4f}" if math.isfinite(m.u_peak_m_s) else "",
                f"{m.t_u_peak_ms:.3f}" if math.isfinite(m.t_u_peak_ms) else "",
                f"{m.aspect_ratio:.4f}" if m.aspect_ratio is not None else "",
                f"{m.we:.2f}" if m.we is not None else "",
                f"{m.re:.1f}" if m.re is not None else "",
            ])
    return out


def _safe(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


# ===========================================================================
# 9. CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render publication-quality figures from "
                    "cv_output.csv result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csvs", nargs="+", type=Path,
                        help="One or more output folders or cv_output.csv files")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Figure output directory "
                             "(default: <csv parent>/figures)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display tags matching the input files, in order")
    parser.add_argument("--d0", choices=("auto", "volume", "area"),
                        default="auto", help="Reference diameter D0 source")
    parser.add_argument("-u", "--impact-velocity", type=float, default=None,
                        metavar="M_S", help="Impact velocity U0 (m/s)")
    parser.add_argument("--density", type=float, default=None,
                        help="Liquid density (kg/m^3), for We / Re")
    parser.add_argument("--surface-tension", type=float, default=None,
                        help="Surface tension (N/m), for We")
    parser.add_argument("--viscosity", type=float, default=None,
                        help="Dynamic viscosity (Pa s), for Re")
    parser.add_argument("--tilt", type=float, default=None,
                        help="Substrate tilt / impact angle (degrees)")
    parser.add_argument("--formats", default="pdf,png",
                        help="Comma-separated export formats: pdf,png,svg,tiff")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Raster export DPI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.csvs = [result_csv_path(path) for path in args.csvs]
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    args.formats = [
        f.strip().lower() for f in args.formats.split(",") if f.strip()
    ]

    for path in args.csvs:
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 2
    if args.labels is not None and len(args.labels) != len(args.csvs):
        print("ERROR: --labels count must match input file count.",
              file=sys.stderr)
        return 2

    apply_style()

    try:
        runs = [load_run(p.absolute()) for p in args.csvs]
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.labels:
        # explicit user labels; keep the input ordering
        for run, label in zip(runs, args.labels):
            run.tag = label
    elif len(runs) > 1:
        # organize the comparison by experimental condition and tag by it
        runs.sort(key=condition_sort_key)
        for run, tag in zip(runs, derive_tags(runs)):
            run.tag = tag

    metrics: list[RunMetrics] = []
    for run in runs:
        d0_mm, d0_source = reference_diameter(run, args.d0)
        metrics.append(
            evaluate_run(
                run,
                d0_mm=d0_mm, d0_source=d0_source,
                impact_velocity=args.impact_velocity,
                rho=args.density, sigma=args.surface_tension,
                mu=args.viscosity,
            )
        )

    if args.output_dir is not None:
        outdir = args.output_dir
    else:
        abs_paths = [p.absolute() for p in args.csvs]
        parents = {p.parent for p in abs_paths}
        if len(parents) == 1:
            base = parents.pop()
        else:
            import os
            base = Path(os.path.commonpath([str(p) for p in parents]))
        outdir = base / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if len(runs) == 1:
        written += figure_single(runs[0], metrics[0], args, outdir)
    else:
        written += figure_comparison(runs, metrics, args, outdir)
        for run, m in zip(runs, metrics):
            written += figure_single(run, m, args, outdir)

    summary = write_summary(runs, metrics, outdir)
    written.append(summary)

    print(f"Wrote {len(written)} files to {outdir}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
