#!/usr/bin/env python3
"""
plot_figures.py — Publication-quality result figures for droplet-impact
spreading measurements.

Standalone script (independent of the droplet_impact_cv analysis workflow).
It reads one or more ``spreading_diameter.csv`` result files produced by the
image-analysis pipeline and renders Nature-style multi-panel figures organized
by physical meaning rather than one plot per variable.

Physics organization
---------------------
Single run  -> ``spreading_dynamics_<tag>``  (asymmetric 4-panel grid)
    a  Spreading factor beta(t) = d/D0          — primary dynamics (hero)
       includes contact-line / rim diameters when they differ, beta_max,
       phase shading (spreading vs recoil), dimensionless top axis tau = tU/D0
    b  Incoming droplet geometry (D_parallel, D_normal, D_eq,V, D_eq,A)
       — setup / initial anisotropy that defines the length scale
    c  Contact-line velocity u_cl(t)            — kinematic mechanism
    d  Lamella morphology: projected area A/A0 and mean thickness h/D0
       — orthogonal shape evidence

Multiple runs -> ``spreading_comparison``
    a  beta(t) overlay with direct labels       — cross-condition dynamics
    b  beta_max bar summary
    c  t_max bar summary

A machine-readable ``spreading_metrics_summary.csv`` is always written.

Inputs
------
Measurements are taken from the current ``spreading_diameter.csv``: time
(time_ms), frame rate (fps), image scale, diameters, impact frame and
component area are read directly from the CSV (or computed from CSV columns);
the script never reads cv_config.txt from the input folder. A
``run_params.json`` in the same result folder supplies settings that are not
stored in the CSV (e.g. the configured surface angle) and is used when
present.

Examples
--------
  uv run python scripts/plot_figures.py outputs/example_tilted_droplet/spreading_diameter.csv
  uv run python scripts/plot_figures.py outputs/run_a/spreading_diameter.csv \
      outputs/run_b/spreading_diameter.csv --labels "We=50" "We=150" \
      --impact-velocity 1.42 --density 998 --surface-tension 0.072 \
      --viscosity 0.001 --tilt 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Loading: current-format CSV is the data source; run_params.json is optional
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ("time_ms", "d_spread_mm")
RUN_PARAMS_FILENAME = "run_params.json"


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


def load_run(path: Path) -> dict:
    """Parse a current-format spreading CSV; run_params.json is optional."""
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
        if name not in headers:
            return np.full(len(rows), np.nan)
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
            params = json.loads(
                params_path.read_text(encoding="utf-8")
            ).get("parameters", {})
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {params_path}: {error}") from None

    run = {
        "path": path,
        "tag": path.parent.name,
        "n_rows": len(rows),
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
        "d_eq_v_px": column("ref_d_eq_volume_px"),
        "d_eq_a_px": column("ref_d_eq_area_px"),
        "area_px2": column("droplet_area_px2"),
        "fps": scalar("fps"),
        "pixel_mm_per_px": scalar("pixel_size_mm_per_px"),
        "impact_frame": scalar("impact_frame"),
        "surface_angle_deg": _to_float(str(params.get("surface_angle_deg", ""))),
    }

    # keep only rows with a valid time axis, sort chronologically
    valid = np.isfinite(run["t_ms"])
    if valid.sum() >= 2:
        order = np.argsort(run["t_ms"][valid], kind="stable")
        for key, val in run.items():
            if isinstance(val, np.ndarray) and val.shape == (len(rows),):
                run[key] = val[valid][order]
        run["n_rows"] = int(valid.sum())
    return run


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def reference_diameter(run: dict, source: str) -> tuple[float, str]:
    """Return (D0 in mm, provenance). source: auto | volume | area."""
    v = run["d_eq_v"]
    a = run["d_eq_a"]
    v = v[np.isfinite(v)][0] if np.isfinite(v).any() else np.nan
    a = a[np.isfinite(a)][0] if np.isfinite(a).any() else np.nan

    if source == "volume" and math.isfinite(v):
        return float(v), "volume-equivalent"
    if source == "area" and math.isfinite(a):
        return float(a), "area-equivalent"
    if source == "auto":
        if math.isfinite(v):
            return float(v), "volume-equivalent"
        if math.isfinite(a):
            return float(a), "area-equivalent"
    # last-resort: largest pre-spread-ish diameter cannot be identified;
    # fall back to the first measured spread diameter only if nothing else.
    d0 = run["d_spread"][0] if run["d_spread"].size else np.nan
    if math.isfinite(d0):
        return float(d0), "first-frame"
    raise ValueError(
        f"No reference diameter could be determined from {run['path']}."
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

        n = mask.sum()
        window = min(7, n if n % 2 == 1 else n - 1)
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


def aspect_ratio(run: dict) -> float | None:
    p = run["d_parallel"]
    n = run["d_normal"]
    if np.isfinite(p).any() and np.isfinite(n).any() and n[0] != 0:
        return float(p[0] / n[0])
    return None


def compute_metrics(run: dict, d0_mm: float, d0_source: str,
                    u0: float | None) -> dict:
    t = run["t_ms"]
    d = run["d_spread"]
    beta = d / d0_mm if math.isfinite(d0_mm) else d.copy()

    finite = np.isfinite(beta)
    idx_max = int(np.nanargmax(beta)) if finite.any() else -1
    beta_max = float(beta[idx_max]) if idx_max >= 0 else math.nan
    t_max = float(t[idx_max]) if idx_max >= 0 else math.nan
    beta_final = float(beta[finite][-1]) if finite.any() else math.nan

    u_cl = smooth_derivative(t / 1000.0, d / 1000.0)  # m/s
    pos = u_cl > 0
    u_peak = float(np.nanmax(u_cl)) if pos.any() else math.nan
    if pos.any():
        t_u_peak = float(t[int(np.nanargmax(np.where(pos, u_cl, np.nan)))])
    else:
        t_u_peak = math.nan

    ar = aspect_ratio(run)

    we = reynolds = None
    if u0 is not None and math.isfinite(d0_mm):
        if run.get("rho") and run.get("sigma"):
            we = run["rho"] * u0**2 * (d0_mm / 1000.0) / run["sigma"]
        if run.get("rho") and run.get("mu"):
            reynolds = run["rho"] * u0 * (d0_mm / 1000.0) / run["mu"]

    return {
        "tag": run["tag"],
        "d0_mm": d0_mm,
        "d0_source": d0_source,
        "beta": beta,
        "beta_max": beta_max,
        "t_max_ms": t_max,
        "beta_final": beta_final,
        "u_cl": u_cl,
        "u_peak_m_s": u_peak,
        "t_u_peak_ms": t_u_peak,
        "aspect_ratio": ar,
        "we": we,
        "re": reynolds,
        "idx_max": idx_max,
    }


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
            bbox_inches="tight", pad_inches=0.04,
        )
        paths.append(target)
    return paths


# ---------------------------------------------------------------------------
# Single-run dynamics figure
# ---------------------------------------------------------------------------

def _condition_lines(run: dict, m: dict, u0: float | None,
                     tilt: float | None) -> list[str]:
    lines = [rf"$D_0$ = {m['d0_mm']:.2f} mm"]
    if u0 is not None:
        lines.append(rf"$U_0$ = {u0:.2f} m s$^{{-1}}$")
    if m["we"] is not None:
        lines.append(rf"$We$ = {m['we']:.0f}")
    if m["re"] is not None:
        lines.append(rf"$Re$ = {m['re']:.0f}")
    if tilt is not None:
        lines.append(rf"$\alpha$ = {tilt:g}$\degree$")
    return lines


def _series_differs(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return False
    scale = np.nanmax(np.abs(a[mask])) or 1.0
    return float(np.nanmax(np.abs(a[mask] - b[mask])) / scale) > 0.01


def figure_single(run: dict, m: dict, args, outdir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = run["t_ms"]
    d0 = m["d0_mm"]
    k = m["idx_max"]
    u0 = args.impact_velocity

    fig = plt.figure(figsize=(7.2, 4.25))
    gs = GridSpec(
        2, 6, figure=fig,
        hspace=0.42, wspace=1.25,
        left=0.065, right=0.985, top=0.93, bottom=0.11,
    )
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:6])

    # --- panel a: spreading factor beta(t) --------------------------------
    ax = ax_a
    ax.plot(t, m["beta"], color=PALETTE["hero"], lw=1.8, marker="o",
            ms=3.4, mfc="white", mew=1.1, label=r"Spreading, $\beta$")

    if _series_differs(run["d_spread"], run["d_contact"]):
        ax.plot(t, run["d_contact"] / d0, color=PALETTE["contact"],
                lw=1.3, ls=(0, (4, 2)), label=r"Contact line")
    if _series_differs(run["d_spread"], run["d_rim"]):
        ax.plot(t, run["d_rim"] / d0, color=PALETTE["rim"],
                lw=1.3, ls=(0, (1.5, 1.5)), label=r"Rim to rim")

    ax.axhline(1.0, color=PALETTE["neutral_mid"], lw=0.7, ls=(0, (3, 3)))
    ax.axvline(m["t_max_ms"], color=PALETTE["neutral_mid"], lw=0.6,
               ls=(0, (2, 3)))

    t_end = float(t[-1])
    if k > 0:
        ax.axvspan(0.0, m["t_max_ms"], color=PALETTE["hero"], alpha=0.06,
                   lw=0)
        ax.text(m["t_max_ms"] / 2, ax.get_ylim()[1] * 0.98,
                "Spreading", ha="center", va="top", fontsize=6.5,
                color=PALETTE["neutral_mid"])
        if t_end > m["t_max_ms"]:
            ax.text((m["t_max_ms"] + t_end) / 2, ax.get_ylim()[1] * 0.98,
                    "Recoil / relaxation", ha="center", va="top",
                    fontsize=6.5, color=PALETTE["neutral_mid"])

    ax.plot(m["t_max_ms"], m["beta_max"], marker="v", ms=5,
            color=PALETTE["hero"], mfc=PALETTE["hero"], zorder=5)
    # peak is a local maximum, so the region directly below it is clear
    ax.annotate(
        rf"$\beta_{{\max}}={m['beta_max']:.2f}$" + "\n"
        + rf"$t={m['t_max_ms']:.2f}$ ms",
        xy=(m["t_max_ms"], m["beta_max"]),
        xytext=(0, -30),
        textcoords="offset points", fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.6,
                        color=PALETTE["neutral_mid"]),
        ha="center",
    )

    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Spreading factor $\beta = d/D_0$")
    ax.set_xlim(left=0)
    ax.locator_params(axis="y", nbins=6)

    # dimensionless top axis when impact velocity is supplied
    if u0 is not None:
        # tau = t_s U / D_m = (t_ms/1000) U / (D_mm/1000) = t_ms U / D_mm
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xbound(ax.get_xbound())
        ticks = ax.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{tk * u0 / d0:.1f}" for tk in ticks])
        ax2.set_xlabel(r"Dimensionless time $\tau = tU_0/D_0$", fontsize=7)
        ax2.spines["top"].set_visible(True)
        ax2.spines["top"].set_linewidth(0.8)
        ax2.tick_params(length=2.0, width=0.7, labelsize=6.5)

    # condition text (top-left inside axes); fall back to the surface angle
    # recorded in run_params.json when --tilt is not supplied
    tilt = args.tilt
    if tilt is None and run.get("surface_angle_deg") is not None:
        tilt = run["surface_angle_deg"]
    info = _condition_lines(run, m, u0, tilt)
    ax.text(
        0.02, 0.97, "\n".join(info), transform=ax.transAxes,
        fontsize=6.6, va="top", ha="left", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75),
    )
    ax.legend(loc="lower right", ncol=1, handlelength=1.8)

    # --- panel b: incoming droplet geometry -------------------------------
    ax = ax_b
    entries: list[tuple[str, float]] = []
    p = run["d_parallel"]
    n = run["d_normal"]
    va = run["d_eq_v"]
    aa = run["d_eq_a"]
    if np.isfinite(p).any():
        entries.append((r"$D_\parallel$", float(p[0])))
    if np.isfinite(aa).any():
        entries.append((r"$D_{\mathrm{eq,A}}$", float(aa[0])))
    if np.isfinite(va).any():
        entries.append((r"$D_{\mathrm{eq,V}}$", float(va[0])))
    if np.isfinite(n).any():
        entries.append((r"$D_\perp$", float(n[0])))

    if entries:
        ys = np.arange(len(entries))[::-1]
        vals = np.array([e[1] for e in entries])
        x_left = float(vals.min() - (vals.max() - vals.min() + 0.3) * 0.6)
        for y, (lab, val) in zip(ys, entries):
            ax.hlines(y, x_left, val, color=PALETTE["neutral_mid"], lw=0.7)
            ax.plot(val, y, marker="o", ms=4, color=PALETTE["hero"])
            ax.text(val + 0.02, y, f"{val:.3f}", va="center", fontsize=6.5)
        ax.set_yticks(ys)
        ax.set_yticklabels([e[0] for e in entries])
        ax.set_xlim(x_left, float(vals.max()) * 1.06)
        ax.set_xlabel("Diameter (mm)")
        title = "Incoming droplet"
        if m["aspect_ratio"] is not None:
            title += "\n" + rf"$D_\parallel/D_\perp={m['aspect_ratio']:.3f}$"
        ax.set_title(title, fontsize=7.3, pad=3)
    else:
        ax.text(0.5, 0.5, "Reference geometry\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=PALETTE["neutral_mid"])
        ax.set_xticks([]); ax.set_yticks([])

    # --- panel c: contact-line velocity -----------------------------------
    ax = ax_c
    u = m["u_cl"]
    if u0 is not None:
        u_plot = u / u0
        ylab = r"Contact-line velocity $u_{\mathrm{cl}}/U_0$"
    else:
        u_plot = u
        ylab = r"Contact-line velocity (m s$^{-1}$)"
    ax.plot(t, u_plot, color=PALETTE["hero"], lw=1.4)
    ax.fill_between(t, u_plot, 0, where=(u_plot > 0),
                    color=PALETTE["up"], alpha=0.10, lw=0)
    ax.fill_between(t, u_plot, 0, where=(u_plot < 0),
                    color=PALETTE["down"], alpha=0.10, lw=0)
    ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.7)
    if np.isfinite(m["u_peak_m_s"]):
        upk = m["u_peak_m_s"] / u0 if u0 is not None else m["u_peak_m_s"]
        ax.plot(m["t_u_peak_ms"], upk, marker="v", ms=4.5,
                color=PALETTE["hero"], mfc=PALETTE["hero"])
        if u0 is not None:
            txt = rf"$u_{{\max}}/U_0={upk:.2f}$"
        else:
            txt = rf"$u_{{\max}}={upk:.2f}$"
        ax.annotate(txt, xy=(m["t_u_peak_ms"], upk),
                    xytext=(6, -4), textcoords="offset points",
                    fontsize=6.6, ha="left")
    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(ylab)
    ax.set_xlim(left=0)

    # --- panel d: lamella morphology --------------------------------------
    ax = ax_d
    px = run["pixel_mm_per_px"]
    d0_px = d0 / px if px else np.nan
    if math.isfinite(d0_px):
        a0_px2 = math.pi / 4.0 * d0_px**2
    else:
        a0_px2 = float(run["area_px2"][0])
    a_ratio = run["area_px2"] / a0_px2

    ax.plot(t, a_ratio, color=PALETTE["morph"], lw=1.5,
            label=r"$A/A_0$")
    ax.fill_between(t, a_ratio, a_ratio.min(), color=PALETTE["morph"],
                    alpha=0.08, lw=0)
    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Projected area $A/A_0$", color=PALETTE["morph"])
    ax.tick_params(axis="y", colors=PALETTE["morph"])
    ax.spines["left"].set_color(PALETTE["morph"])
    ax.set_xlim(left=0)

    # mean lamella thickness h = A_physical / d_spread
    if px:
        a_mm2 = run["area_px2"] * px**2
        with np.errstate(divide="ignore", invalid="ignore"):
            h_mm = np.where(run["d_spread"] > 0.5 * d0,
                            a_mm2 / run["d_spread"], np.nan)
        h_norm = h_mm / d0
        ax2 = ax.twinx()
        ax2.plot(t, h_norm, color=PALETTE["thickness"], lw=1.2,
                 ls=(0, (3, 2)), label=r"$h/D_0$")
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

    i_min = int(np.nanargmin(a_ratio))
    ax.plot(t[i_min], a_ratio[i_min], marker="o", ms=4,
            color=PALETTE["morph"])
    ax.annotate(rf"max flattening" + "\n"
                + rf"$A/A_0={a_ratio[i_min]:.2f}$",
                xy=(t[i_min], a_ratio[i_min]),
                xytext=(8, 12), textcoords="offset points", fontsize=6.6,
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=PALETTE["neutral_mid"]))

    for ax_, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax_, lab)

    stem = f"spreading_dynamics_{_safe(run['tag'])}"
    return save_figure(fig, outdir, stem, args.formats, args.dpi)


# ---------------------------------------------------------------------------
# Multi-run comparison figure
# ---------------------------------------------------------------------------

def figure_comparison(runs: list[dict], metrics: list[dict],
                      args, outdir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(
        2, 6, figure=fig,
        hspace=0.95, wspace=1.3,
        left=0.07, right=0.98, top=0.93, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[:, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 4:6])

    tags = [r["tag"] for r in runs]
    colors = [RUN_COLORS[i % len(RUN_COLORS)] for i in range(len(runs))]

    # --- panel a: beta(t) overlay with direct labels ----------------------
    ax = ax_a
    label_targets: list[tuple[float, float, str, str]] = []
    for run, m, color in zip(runs, metrics, colors):
        t = run["t_ms"]
        ax.plot(t, m["beta"], color=color, lw=1.6, label=m["tag"])
        ax.plot(m["t_max_ms"], m["beta_max"], marker="v", ms=4.5,
                color=color, mfc=color)
        label_targets.append((float(t[-1]), float(m["beta"][-1]),
                              m["tag"], color))

    ax.axhline(1.0, color=PALETTE["neutral_mid"], lw=0.7, ls=(0, (3, 3)))
    ax.set_xlabel("Time after impact (ms)")
    ax.set_ylabel(r"Spreading factor $\beta = d/D_0$")
    ax.set_xlim(left=0)

    t_max_all = max(tg[0] for tg in label_targets)
    ax.set_xlim(0, t_max_all * 1.24)

    # greedy de-collision of direct end labels
    yspan = float(ax.get_ylim()[1] - ax.get_ylim()[0])
    min_dy = 0.075 * yspan
    placed: list[tuple[float, float]] = []
    for x, y, tag, color in sorted(label_targets, key=lambda z: z[1]):
        y_adj = y
        for px_, py_ in placed:
            if abs(y_adj - py_) < min_dy:
                y_adj = py_ + min_dy
        placed.append((x, y_adj))
        ax.plot([x, t_max_all * 1.03], [y, y_adj], color=color,
                lw=0.5, ls=(0, (2, 2)))
        ax.text(t_max_all * 1.045, y_adj, tag, color=color,
                fontsize=7, va="center", ha="left")
    ax.set_ylim(top=float(ax.get_ylim()[1]) * 1.06)

    # --- panel b: beta_max bars -------------------------------------------
    ax = ax_b
    x = np.arange(len(runs))
    bmax = [m["beta_max"] for m in metrics]
    ax.bar(x, bmax, color=colors, width=0.66, lw=0)
    for xi, v in zip(x, bmax):
        ax.text(xi, v + max(bmax) * 0.02, f"{v:.2f}", ha="center",
                va="bottom", fontsize=6.5)
    ax.axhline(1.0, color=PALETTE["neutral_mid"], lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=6.3)
    ax.set_ylabel(r"$\beta_{\max}$", fontsize=7.5)
    ax.set_ylim(0, max(bmax) * 1.16)
    ax.set_title("Maximum spreading", fontsize=7.5)

    # --- panel c: t_max bars ----------------------------------------------
    ax = ax_c
    tmax = [m["t_max_ms"] for m in metrics]
    ax.bar(x, tmax, color=colors, width=0.66, lw=0, alpha=0.85)
    for xi, v in zip(x, tmax):
        ax.text(xi, v + max(tmax) * 0.02, f"{v:.2f}", ha="center",
                va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=6.3)
    ax.set_ylabel(r"$t_{\max}$ (ms)", fontsize=7.5)
    ax.set_ylim(0, max(tmax) * 1.16)
    ax.set_title("Time to maximum", fontsize=7.5)

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")

    return save_figure(fig, outdir, "spreading_comparison",
                       args.formats, args.dpi)


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def write_summary(runs: list[dict], metrics: list[dict],
                  outdir: Path) -> Path:
    out = outdir / "spreading_metrics_summary.csv"
    cols = [
        "tag", "n_frames", "duration_ms", "d0_mm", "d0_source",
        "beta_max", "t_max_ms", "beta_final", "u_peak_m_s",
        "t_u_peak_ms", "aspect_ratio", "weber_number", "reynolds_number",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for run, m in zip(runs, metrics):
            writer.writerow([
                m["tag"], run["n_rows"],
                f"{float(run['t_ms'][-1]):.3f}",
                f"{m['d0_mm']:.4f}", m["d0_source"],
                f"{m['beta_max']:.4f}", f"{m['t_max_ms']:.3f}",
                f"{m['beta_final']:.4f}",
                f"{m['u_peak_m_s']:.4f}" if math.isfinite(m["u_peak_m_s"]) else "",
                f"{m['t_u_peak_ms']:.3f}" if math.isfinite(m["t_u_peak_ms"]) else "",
                f"{m['aspect_ratio']:.4f}" if m["aspect_ratio"] is not None else "",
                f"{m['we']:.2f}" if m["we"] is not None else "",
                f"{m['re']:.1f}" if m["re"] is not None else "",
            ])
    return out


def _safe(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render publication-quality figures from "
                    "spreading_diameter.csv + run_params.json result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csvs", nargs="+", type=Path,
                   help="One or more spreading_diameter.csv files")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Figure output directory (default: <csv parent>/figures)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display tags matching the input files, in order")
    p.add_argument("--d0", choices=("auto", "volume", "area"), default="auto",
                   help="Reference diameter D0 source")
    p.add_argument("-u", "--impact-velocity", type=float, default=None,
                   metavar="M_S", help="Impact velocity U0 (m/s)")
    p.add_argument("--density", type=float, default=None,
                   help="Liquid density (kg/m^3), for We / Re")
    p.add_argument("--surface-tension", type=float, default=None,
                   help="Surface tension (N/m), for We")
    p.add_argument("--viscosity", type=float, default=None,
                   help="Dynamic viscosity (Pa s), for Re")
    p.add_argument("--tilt", type=float, default=None,
                   help="Substrate tilt / impact angle (degrees)")
    p.add_argument("--formats", default="pdf,png",
                   help="Comma-separated export formats: pdf,png,svg,tiff")
    p.add_argument("--dpi", type=int, default=600,
                   help="Raster export DPI")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    # validate inputs
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
        for run, label in zip(runs, args.labels):
            run["tag"] = label

    # shared physical properties
    for run in runs:
        run["rho"] = args.density
        run["sigma"] = args.surface_tension
        run["mu"] = args.viscosity

    metrics: list[dict] = []
    for run in runs:
        d0_mm, d0_src = reference_diameter(run, args.d0)
        metrics.append(compute_metrics(run, d0_mm, d0_src,
                                       args.impact_velocity))

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
