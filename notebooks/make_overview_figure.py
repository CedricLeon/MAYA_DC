"""make_overview_figure.py — standalone script to produce the paper overview figure.

Run from the repo root (or any directory):
    python notebooks/make_overview_figure.py

REQUIRES the raw reconstruction data (per-lambda ``*_phys.npy`` + the ``*_metrics.csv``),
kept off-git under ``data/from_cluster/results_extracted/`` — see ``docs/raw-data.md``.

Outputs (regenerable, written off-git):
    data/from_cluster/overview_figure.{pdf,svg}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display needed
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root + paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


def find_project_root(start: Path, marker: str = ".project-root") -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find project root from {start}.")


PROJECT_ROOT = find_project_root(_HERE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "from_cluster" / "results_extracted" / "results"

# ---------------------------------------------------------------------------
# Config — edit here to iterate
# ---------------------------------------------------------------------------
LAMBDAS = [3, 10, 20, 1000]  # columns
SEED = 0
CLIP = 3.0
CELL_W = 2.5  # inches per column
CELL_H = 2.5  # inches per row
# Row-0 (originals) vertical scale relative to reconstruction rows
ROW0_H_RATIO = 1.0
OUTPUT_STEM = PROJECT_ROOT / "data" / "from_cluster" / "overview_figure"

# ---------------------------------------------------------------------------
# Inline helpers (copy of the primitives from viewer_helpers.py so this script
# is completely self-contained and you can edit it freely without touching the
# shared library).
# ---------------------------------------------------------------------------
EPS = 1e-8


def phys_to_logI(phys: np.ndarray, eps: float = EPS) -> np.ndarray:
    if phys.ndim == 3 and phys.shape[0] == 2:
        re, im = phys[0], phys[1]
    else:
        raise ValueError(f"Unexpected phys shape: {phys.shape}")
    return np.log(re.astype(np.float64) ** 2 + im.astype(np.float64) ** 2 + eps).astype(np.float32)


def clip_logI(logI: np.ndarray, clip_factor: float = 3.0) -> tuple[float, float]:
    mu, sigma = float(logI.mean()), float(logI.std())
    return mu - clip_factor * sigma, mu + clip_factor * sigma


def load_phys(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr
    raise ValueError(f"Unexpected shape {arr.shape} in {path}")


def show_panel(
    ax: plt.Axes,
    phys: np.ndarray,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation: str | None = None,
) -> None:
    """Render one SAR panel in log-intensity, no axes, no colorbar."""
    logI = phys_to_logI(phys)
    _vmin = vmin if vmin is not None else clip_logI(logI, CLIP)[0]
    _vmax = vmax if vmax is not None else clip_logI(logI, CLIP)[1]
    ax.imshow(logI, cmap="viridis", origin="upper", aspect="equal", vmin=_vmin, vmax=_vmax)
    ax.set_facecolor("black")
    ax.axis("off")
    ax.grid(False)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=2)
    if annotation:
        ax.text(
            0.5,
            -0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            ha="center",
            va="top",
            color="black",
            linespacing=1.35,
        )


def fmt_lambda(lmbda: int) -> str:
    """'λ = 15' using the Unicode Greek lambda."""
    return f"λ = {lmbda}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
_RECON_RE = re.compile(
    r"^fixed_patch_(?P<product_id>.+?)"
    r"_(?P<variant>sq|nsq)_lambda(?P<lmbda>\d+)_seed(?P<seed>\d+)"
    r"_(?P<domain>rcmc|slc)_recon_phys\.npy$"
)
_ORIG_RE = re.compile(
    r"^fixed_patch_(?P<product_id>.+?)" r"_(?P<domain>rcmc|slc)_original_phys\.npy$"
)

recon_index: list[dict[str, Any]] = []
originals: dict[tuple[str, str], Path] = {}

for npy_path in sorted(RESULTS_DIR.glob("*.npy")):
    m = _RECON_RE.match(npy_path.name)
    if m:
        recon_index.append(
            {
                "product_id": m.group("product_id"),
                "variant": m.group("variant"),
                "lmbda": int(m.group("lmbda")),
                "seed": int(m.group("seed")),
                "domain": m.group("domain"),
                "phys_path": npy_path,
            }
        )
        continue
    m = _ORIG_RE.match(npy_path.name)
    if m:
        originals[(m.group("product_id"), m.group("domain"))] = npy_path

if not recon_index:
    raise RuntimeError(f"No _phys.npy reconstruction files found in {RESULTS_DIR}")

file_index_df = pd.DataFrame(recon_index)
product_id = file_index_df["product_id"].iloc[0]

# Metrics CSV
metrics_csvs = sorted(RESULTS_DIR.glob("*_sq_nsq_seed*_metrics.csv"))
if not metrics_csvs:
    raise FileNotFoundError(f"No metrics CSV found in {RESULTS_DIR}")
metrics_df = pd.read_csv(metrics_csvs[0])
print(f"Product: {product_id}")
print(f"Lambdas requested: {LAMBDAS}")
print(f"Metrics CSV: {metrics_csvs[0].name}")


def _lookup(variant: str, lmbda: int, domain: str) -> Path:
    rows = file_index_df[
        (file_index_df["variant"] == variant)
        & (file_index_df["lmbda"] == lmbda)
        & (file_index_df["seed"] == SEED)
        & (file_index_df["domain"] == domain)
    ]
    if rows.empty:
        raise KeyError(f"variant={variant} lmbda={lmbda} domain={domain} seed={SEED}")
    return rows.iloc[0]["phys_path"]


def _metrics_str(variant: str, lmbda: int) -> str:
    rows = metrics_df[
        (metrics_df["variant"] == variant)
        & (metrics_df["lambda"] == lmbda)
        & (metrics_df["seed"] == SEED)
    ]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    parts = []
    if "bpp" in row and pd.notna(row["bpp"]):
        parts.append(f"bpp={float(row['bpp']):.3f}")
    if "slc_psnr_amp" in row and pd.notna(row["slc_psnr_amp"]):
        parts.append(f"PSNR={float(row['slc_psnr_amp']):.1f} dB")
    if "slc_ssim_amp" in row and pd.notna(row["slc_ssim_amp"]):
        parts.append(f"SSIM={float(row['slc_ssim_amp']):.3f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Load originals + compute shared intensity scales
# ---------------------------------------------------------------------------
rcmc_orig = (
    load_phys(originals[(product_id, "rcmc")]) if (product_id, "rcmc") in originals else None
)
slc_orig = load_phys(originals[(product_id, "slc")]) if (product_id, "slc") in originals else None

rcmc_vmin, rcmc_vmax = (
    clip_logI(phys_to_logI(rcmc_orig), CLIP) if rcmc_orig is not None else (None, None)
)
slc_vmin, slc_vmax = (
    clip_logI(phys_to_logI(slc_orig), CLIP) if slc_orig is not None else (None, None)
)

n = len(LAMBDAS)

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
# Layout: 3 rows x (2n) cols.  A thin black divider line is drawn in figure
# coordinates between column n-1 and column n.
#
# Row 0  — originals: RCMC original | SLC original
#           Each original spans ALL n columns of its block so they are centred.
#           We achieve this with a sub-gridspec.
# Row 1  — SQ: n RCMC recons | n SLC recons
# Row 2  — NSQ: n RCMC recons | n SLC recons

fig_w = CELL_W * 2 * n
# Row heights: row-0 scaled, rows 1-2 uniform
row0_h = CELL_H * ROW0_H_RATIO
fig_h = row0_h + CELL_H * 2  # extra for row labels / suptitle

fig = plt.figure(figsize=(fig_w, fig_h))

# Height ratios: row0 may differ from rows 1-2
h_ratios = [ROW0_H_RATIO, 1.0, 1.0]
outer_gs = fig.add_gridspec(
    3,
    1,
    hspace=0.15,
    height_ratios=h_ratios,
    left=0.06,
    right=0.99,
    top=0.99,
    bottom=0.04,
)

row0_gs = outer_gs[0].subgridspec(1, 2, wspace=0.02)
row1_gs = outer_gs[1].subgridspec(1, 2 * n, wspace=0.02)
row2_gs = outer_gs[2].subgridspec(1, 2 * n, wspace=0.02)

# --- Row 0: originals ---------------------------------------------------
ax_rcmc_orig = fig.add_subplot(row0_gs[0, 0])
ax_slc_orig = fig.add_subplot(row0_gs[0, 1])

if rcmc_orig is not None:
    show_panel(ax_rcmc_orig, rcmc_orig, title=None, vmin=rcmc_vmin, vmax=rcmc_vmax)
else:
    ax_rcmc_orig.set_visible(False)

if slc_orig is not None:
    show_panel(ax_slc_orig, slc_orig, title=None, vmin=slc_vmin, vmax=slc_vmax)
else:
    ax_slc_orig.set_visible(False)

# --- Rows 1-2: SQ then NSQ ----------------------------------------------
ROW_LABELS = {0: "REFERENCE", 1: "SH", 2: "SH - NSK"}

for row_idx, variant in enumerate(["original", "sq", "nsq"], start=0):
    row_gs = row1_gs if row_idx == 1 else row2_gs

    # Left row label (rotated, outside the axes)
    # Place it at the vertical midpoint of the current outer row.
    outer_bbox = outer_gs[row_idx].get_position(fig)
    label_y = (outer_bbox.y0 + outer_bbox.y1) / 2
    fig.text(
        0.05,
        label_y,
        ROW_LABELS[row_idx],
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=90,
        color="0.2",
    )

    # RCMC recons (left n columns)
    for col_idx, lmbda in enumerate(LAMBDAS):
        ax = fig.add_subplot(row_gs[0, col_idx])
        try:
            phys = load_phys(_lookup(variant, lmbda, "rcmc"))
            v0 = rcmc_vmin if rcmc_vmin is not None else clip_logI(phys_to_logI(phys), CLIP)[0]
            v1 = rcmc_vmax if rcmc_vmax is not None else clip_logI(phys_to_logI(phys), CLIP)[1]
            show_panel(ax, phys, title=fmt_lambda(lmbda), vmin=v0, vmax=v1)
        except KeyError as exc:
            ax.set_visible(False)
            print(f"[WARN] {exc}")

    # SLC recons (right n columns) with metrics annotation
    for col_idx, lmbda in enumerate(LAMBDAS):
        ax = fig.add_subplot(row_gs[0, n + col_idx])
        try:
            phys = load_phys(_lookup(variant, lmbda, "slc"))
            annot = _metrics_str(variant, lmbda)
            v0 = slc_vmin if slc_vmin is not None else clip_logI(phys_to_logI(phys), CLIP)[0]
            v1 = slc_vmax if slc_vmax is not None else clip_logI(phys_to_logI(phys), CLIP)[1]
            show_panel(ax, phys, title=fmt_lambda(lmbda), vmin=v0, vmax=v1, annotation=annot)
        except KeyError as exc:
            ax.set_visible(False)
            print(f"[WARN] {exc}")

# ---------------------------------------------------------------------------
# RCMC / SLC block labels + vertical divider line
# ---------------------------------------------------------------------------
# We need the figure-coordinate x position of the boundary between the RCMC
# and SLC halves.  We derive it from the gridspec geometry after drawing.
fig.canvas.draw()  # force layout so get_position() is accurate

# The divider sits between the last RCMC column and first SLC column of row 1.
ax_last_rcmc = fig.axes[2 + n - 1]  # row0 uses 2 axes; row1 starts at index 2
ax_first_slc = fig.axes[2 + n]

bbox_l = ax_last_rcmc.get_position()
bbox_r = ax_first_slc.get_position()
divider_x = (bbox_l.x1 + bbox_r.x0) / 2.0

# Vertical divider line spanning rows 0 and 2
outer_r0 = outer_gs[0].get_position(fig)
outer_r1 = outer_gs[1].get_position(fig)
outer_r2 = outer_gs[2].get_position(fig)
line_y0 = outer_r2.y0
line_y1 = outer_r1.y1
divider_line = mlines.Line2D(
    [divider_x, divider_x],
    [line_y0, line_y1],
    transform=fig.transFigure,
    color="black",
    linewidth=1.2,
    linestyle="-",
    clip_on=False,
)
fig.add_artist(divider_line)

# "RCMC" and "SLC" block labels just above row 0, centered above the original image
label_y = outer_r0.y1 + 0.015
rcmc_panel = row0_gs[0, 0].get_position(fig)
slc_panel = row0_gs[0, 1].get_position(fig)
label_x_rcmc = (rcmc_panel.x0 + rcmc_panel.x1) / 2
label_x_slc = (slc_panel.x0 + slc_panel.x1) / 2

fig.text(
    label_x_rcmc,
    label_y,
    "RCMC (input)",
    ha="center",
    va="bottom",
    fontsize=13,
    fontweight="bold",
    color="0.15",
)
fig.text(
    label_x_slc,
    label_y,
    "SLC (objective)",
    ha="center",
    va="bottom",
    fontsize=13,
    fontweight="bold",
    color="0.15",
)

# Arrow from RCMC original to SLC original, drawn in the divider gap
bbox_rcmc0 = ax_rcmc_orig.get_position()
bbox_slc0 = ax_slc_orig.get_position()
arrow_y = (bbox_rcmc0.y0 + bbox_rcmc0.y1) / 2
arrow_x0 = bbox_rcmc0.x1 + 0.13
arrow_x1 = bbox_slc0.x0 - 0.13
arrow_xc = (arrow_x0 + arrow_x1) / 2

arrow = mpatches.FancyArrowPatch(
    posA=(arrow_x0, arrow_y),
    posB=(arrow_x1, arrow_y),
    transform=fig.transFigure,
    arrowstyle="-|>",
    color="0.2",
    linewidth=1.5,
    mutation_scale=12,
    clip_on=False,
)
fig.add_artist(arrow)
fig.text(
    arrow_xc,
    arrow_y + 0.01,
    "azimuth compression",
    ha="center",
    va="bottom",
    fontsize=9,
    color="0.2",
    linespacing=1.3,
)

# ---------------------------------------------------------------------------
# Save overview figure
# ---------------------------------------------------------------------------
OUTPUT_DIR = OUTPUT_STEM.parent / "overview_reconstructions"
OUTPUT_DIR.mkdir(exist_ok=True)

for ext in ["pdf", "png"]:
    out = OUTPUT_DIR / OUTPUT_STEM.with_suffix(f".{ext}").name
    dpi = 300 if ext == "png" else 150
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    print(f"Saved {out}")

plt.close(fig)

# ---------------------------------------------------------------------------
# Save individual panels
# ---------------------------------------------------------------------------
PANEL_DPI = 300
PANEL_FIGSIZE = (CELL_W, CELL_H)


def save_single_panel(
    phys: np.ndarray,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save one SAR patch as a standalone borderless PNG."""
    logI = phys_to_logI(phys)
    _vmin = vmin if vmin is not None else clip_logI(logI, CLIP)[0]
    _vmax = vmax if vmax is not None else clip_logI(logI, CLIP)[1]
    fig_s, ax_s = plt.subplots(1, 1, figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    ax_s.imshow(logI, cmap="viridis", origin="upper", aspect="auto", vmin=_vmin, vmax=_vmax)
    ax_s.set_facecolor("black")
    ax_s.axis("off")
    fig_s.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_s.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig_s)
    print(f"  {out_path.name}")


print("\nSaving individual panels...")

# Originals
if rcmc_orig is not None:
    save_single_panel(rcmc_orig, OUTPUT_DIR / "original_rcmc.png", vmin=rcmc_vmin, vmax=rcmc_vmax)
if slc_orig is not None:
    save_single_panel(slc_orig, OUTPUT_DIR / "original_slc.png", vmin=slc_vmin, vmax=slc_vmax)

# Reconstructions
for variant in ["sq", "nsq"]:
    for lmbda in LAMBDAS:
        for domain, (vmin_d, vmax_d) in [
            ("rcmc", (rcmc_vmin, rcmc_vmax)),
            ("slc", (slc_vmin, slc_vmax)),
        ]:
            try:
                phys = load_phys(_lookup(variant, lmbda, domain))
                fname = f"{variant}_lambda{lmbda:04d}_{domain}_recon.png"
                save_single_panel(phys, OUTPUT_DIR / fname, vmin=vmin_d, vmax=vmax_d)
            except KeyError as exc:
                print(f"  [SKIP] {exc}")

# ---------------------------------------------------------------------------
# Metrics report (TXT)
# ---------------------------------------------------------------------------
metrics_lines: list[str] = []

for variant in ["sq", "nsq"]:
    variant_label = "SQ" if variant == "sq" else "NSQ"
    metrics_lines.append(f"=== {variant_label} ===")

    for lmbda in LAMBDAS:
        rows = metrics_df[
            (metrics_df["variant"] == variant)
            & (metrics_df["lambda"] == lmbda)
            & (metrics_df["seed"] == SEED)
        ]
        if rows.empty:
            metrics_lines.append(f"lambda={lmbda}  [no data]")
            metrics_lines.append("")
            continue

        row = rows.iloc[0]

        def _v(col: str) -> float:
            return float(row[col]) if col in row and pd.notna(row[col]) else float("nan")

        bpp = _v("bpp")
        metrics_lines.append(f"lambda={lmbda}")
        metrics_lines.append(f"bpp={bpp:.3f}")

        for domain in ["rcmc", "slc"]:
            psnr = _v(f"{domain}_psnr_amp")
            ssim = _v(f"{domain}_ssim_amp")
            corr = _v(f"{domain}_complex_corr_mean")
            pherr = _v(f"{domain}_phase_err_mean")
            metrics_lines.append(f"--- {domain.upper()} ---")
            metrics_lines.append(f"PSNR={psnr:.1f}dB, SSIM={ssim:.2f}")
            metrics_lines.append(f"corr={corr:.2f}, ph_err={pherr:.2f}")

        metrics_lines.append("")

    metrics_lines.append("")

metrics_txt_path = OUTPUT_DIR / "metrics.txt"
metrics_txt_path.write_text("\n".join(metrics_lines))
print(f"\nSaved metrics report -> {metrics_txt_path}")
