r"""Standalone data visualisation script (F13).

Loads patches through **three independent paths** to diagnose any unexpected
operations in the data pipeline:

  1. **DataModule** — exact same ``MAYA4DataModule`` pipeline used in training.
  2. **Manual** — zarr → real/imag split → ``minmax_normalize`` by hand.
     Should be *identical* to path 1 after the BUG 27 fix.  Any discrepancy
     reveals an unexpected operation inside the MAYA4 library.
  3. **Raw** — zarr complex128, real/imag split, NO normalization.
     Shows the physical value range actually stored in the zarr.

Figures produced:

  - ``channel_distributions.png`` — 3-row x 4-col histogram comparison.
  - ``logI_images.png``           — 3-row x n-col log-intensity image grid.

.. note::
   **SAR visualization convention** (enforced throughout this script):

   * Display: always **log-intensity** ``logI = ln(re² + im² + ε)``,
     clipped to ``mean ± clip_factor·std``.
   * Metrics: always **linear amplitude** ``|·| = sqrt(re² + im²)``.
     Never compute quality metrics (PSNR, SSIM, coherence) on log-scale data.

Usage::

    python scripts/visualize_data.py
    python scripts/visualize_data.py --parts PT4 --n_patches 6
    python scripts/visualize_data.py --parts PT1 --patch_size 256 --buffer 128
    python scripts/visualize_data.py --parts PT4 --patch_size 512 --buffer 512 \\
        --n_patches 4 --max_products 2 --output_dir logs/visualizations/
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

matplotlib.use("Agg")  # no display needed — saves to file

# ---------------------------------------------------------------------------
# Path setup — works regardless of cwd
# ---------------------------------------------------------------------------
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from maya4 import GT_MAX, GT_MIN, RC_MAX, RC_MIN, minmax_inverse, minmax_normalize

from src.data.maya4_datamodule import MAYA4DataModule
from src.utils.logging import print_images_statistics
from src.utils.processing_utils import (
    EPS,
    clip_mean_std_numpy,
    phys_to_linA_torch,
    phys_to_logI_torch,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAYA4 data-pipeline visualisation (F13)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="data/", help="Root data folder with zarr products")
    p.add_argument(
        "--parts",
        default="PT4",
        help="Comma-separated geographic partition(s), e.g. 'PT4' or 'PT1,PT2'",
    )
    p.add_argument(
        "--patch_size", type=int, default=512, help="Core patch size in azimuth AND range"
    )
    p.add_argument("--buffer", type=int, default=512, help="Azimuth buffer lines on each side")
    p.add_argument("--n_patches", type=int, default=4, help="Number of patches to visualise")
    p.add_argument("--max_products", type=int, default=2, help="Max zarr products to load")
    p.add_argument(
        "--output_dir",
        default="logs/visualizations/",
        help="Directory to save the PNG figures",
    )
    p.add_argument("--online", action="store_true", help="Enable HuggingFace streaming")
    p.add_argument(
        "--clip_factor",
        type=float,
        default=3.0,
        help="Std-dev multiplier for contrast clipping in image panels",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_normalised_batch(args: argparse.Namespace):
    """Load one batch through the exact MAYA4DataModule training pipeline."""
    parts = [p.strip() for p in args.parts.split(",")]

    dm = MAYA4DataModule(
        data_dir=args.data_dir,
        patch_size=(args.patch_size, args.patch_size),
        azimuth_buffer=args.buffer,
        batch_size=args.n_patches,
        num_workers=0,
        pin_memory=False,
        train_parts=parts,
        val_parts=parts,
        test_parts=parts,
        max_products_train=args.max_products,
        max_products_val=args.max_products,
        max_products_test=args.max_products,
        samples_per_prod=max(args.n_patches * 2, 20),
        online=args.online,
    )
    dm.setup("fit")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    rcmc, slc, _meta, _eph, coords = batch

    # Clamp to requested n_patches (batch might be smaller if few products)
    n = min(args.n_patches, rcmc.shape[0])
    return rcmc[:n], slc[:n], coords[:n]


def load_manual_pipeline_batch(
    coords: list,
    patch_h: int,
    patch_w: int,
) -> tuple:
    """Load the same coordinates as the DataModule, but manually.

    Steps: zarr → complex128 → real/imag split → ``minmax_normalize``.
    This mirrors exactly what we *expect* the MAYA4 library to do after the
    BUG 27 fix (split BEFORE normalize).  Comparing these tensors against the
    DataModule output reveals any remaining unexpected library operation.

    Returns:
        Tuple ``(rcmc, slc)`` of shape ``(B, 2, patch_h, patch_w)`` float32,
        values in ``[0, 1]`` (same range as DataModule output).
    """
    rcmc_list, slc_list = [], []
    for c in coords:
        zfile, y, x = c["zfile"], int(c["y"]), int(c["x"])
        store = zarr.open(zfile, mode="r")
        rcmc_cplx = store["rcmc"][y : y + patch_h, x : x + patch_w]  # complex128
        slc_cplx = store["az"][y : y + patch_h, x : x + patch_w]  # complex128

        # Split BEFORE normalize (mirrors BUG 27 fix in the library)
        rcmc_ri = np.stack([np.real(rcmc_cplx), np.imag(rcmc_cplx)], axis=-1).astype(
            np.float64
        )  # (H,W,2)
        slc_ri = np.stack([np.real(slc_cplx), np.imag(slc_cplx)], axis=-1).astype(np.float64)

        # Apply NormalizationModule manually: (x − min) / (max − min)
        rcmc_norm = minmax_normalize(rcmc_ri, RC_MIN, RC_MAX).astype(np.float32)
        slc_norm = minmax_normalize(slc_ri, GT_MIN, GT_MAX).astype(np.float32)

        # Permute (H,W,2) → (2,H,W) and add batch dim
        rcmc_list.append(torch.from_numpy(rcmc_norm.transpose(2, 0, 1)).unsqueeze(0))
        slc_list.append(torch.from_numpy(slc_norm.transpose(2, 0, 1)).unsqueeze(0))

        del rcmc_cplx, slc_cplx, rcmc_ri, slc_ri, rcmc_norm, slc_norm
        gc.collect()

    return torch.cat(rcmc_list, dim=0), torch.cat(slc_list, dim=0)


def load_raw_zarr_batch(
    coords: list,
    patch_h: int,
    patch_w: int,
) -> tuple:
    """Load raw physical-scale data: zarr → real/imag split, NO normalization.

    Returns:
        Tuple ``(rcmc, slc)`` of shape ``(B, 2, patch_h, patch_w)`` float32
        in the original physical range (±3 000 for RCMC, ±12 000 for SLC).
    """
    rcmc_list, slc_list = [], []
    for c in coords:
        zfile, y, x = c["zfile"], int(c["y"]), int(c["x"])
        store = zarr.open(zfile, mode="r")
        rcmc_cplx = store["rcmc"][y : y + patch_h, x : x + patch_w]
        slc_cplx = store["az"][y : y + patch_h, x : x + patch_w]

        # (2, H, W) physical float32 — no normalization
        rcmc_ri = np.stack([np.real(rcmc_cplx), np.imag(rcmc_cplx)], axis=0).astype(np.float32)
        slc_ri = np.stack([np.real(slc_cplx), np.imag(slc_cplx)], axis=0).astype(np.float32)

        rcmc_list.append(torch.from_numpy(rcmc_ri).unsqueeze(0))
        slc_list.append(torch.from_numpy(slc_ri).unsqueeze(0))

        del rcmc_cplx, slc_cplx, rcmc_ri, slc_ri
        gc.collect()

    return torch.cat(rcmc_list, dim=0), torch.cat(slc_list, dim=0)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _batch_to_channel_dict(
    rcmc: torch.Tensor,
    slc: torch.Tensor,
    prefix: str,
) -> dict:
    """Build a ``{name: flat_array}`` dict for ``print_images_statistics``."""
    return {
        f"{prefix} RCMC re": rcmc[:, 0].numpy().ravel(),
        f"{prefix} RCMC im": rcmc[:, 1].numpy().ravel(),
        f"{prefix} SLC  re": slc[:, 0].numpy().ravel(),
        f"{prefix} SLC  im": slc[:, 1].numpy().ravel(),
    }


# ---------------------------------------------------------------------------
# Figure 1 — Channel distributions
# ---------------------------------------------------------------------------


def plot_channel_distributions(
    rcmc_dm: torch.Tensor,
    slc_dm: torch.Tensor,
    rcmc_manual: torch.Tensor,
    slc_manual: torch.Tensor,
    rcmc_raw: torch.Tensor,
    slc_raw: torch.Tensor,
    output_path: Path,
    n_bins: int = 80,
) -> None:
    """3-row × 4-col histogram grid comparing the three loading paths.

    Rows:
      0. DataModule output  — what the model actually trains on
      1. Manual pipeline    — zarr → split → normalize (expected equivalent)
      2. Raw physical       — unprocessed data from zarr

    Columns: RCMC re | RCMC im | SLC re | SLC im

    Rows 0 and 1 should be *identical*; any visible difference flags a
    pipeline inconsistency.
    """
    ROW_DEFS = [
        (rcmc_dm, slc_dm, "DataModule\n(training path)"),
        (rcmc_manual, slc_manual, "Manual\n(zarr→split→normalize)"),
        (rcmc_raw, slc_raw, "Raw physical\n(no normalization)"),
    ]
    CHANNELS = [
        ("RCMC", "real", 0, RC_MIN, RC_MAX, "royalblue"),
        ("RCMC", "imag", 1, RC_MIN, RC_MAX, "cornflowerblue"),
        ("SLC", "real", 0, GT_MIN, GT_MAX, "darkorange"),
        ("SLC", "imag", 1, GT_MIN, GT_MAX, "peru"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        "Channel value distributions — three loading paths\n"
        "Rows 0 & 1 must match if the normalization pipeline is correct",
        fontsize=12,
    )

    for row_idx, (rcmc_t, slc_t, row_label) in enumerate(ROW_DEFS):
        is_raw = row_idx == 2
        axes[row_idx, 0].set_ylabel(row_label, fontsize=9, labelpad=8)

        for col_idx, (src, ch_name, ch_idx, vmin, vmax, color) in enumerate(CHANNELS):
            ax = axes[row_idx, col_idx]
            arr = (rcmc_t if src == "RCMC" else slc_t)[:, ch_idx].numpy().ravel()

            ax.hist(arr, bins=n_bins, color=color, alpha=0.8, density=True)

            if not is_raw:
                ax.axvline(0.0, color="k", lw=0.8, ls="--")
                ax.axvline(1.0, color="k", lw=0.8, ls="--")
                ax.set_xlabel("normalised value")
                fmt = ".4f"
            else:
                ax.axvline(vmin, color="k", lw=0.8, ls="--", label=f"norm min ({vmin})")
                ax.axvline(vmax, color="k", lw=0.8, ls="--", label=f"norm max ({vmax})")
                ax.set_xlabel("physical value")
                ax.legend(fontsize=7)
                fmt = ".0f"

            ax.set_ylabel("density")
            mu, sigma = arr.mean(), arr.std()
            ax.text(
                0.97,
                0.97,
                f"μ={mu:{fmt}}\nσ={sigma:{fmt}}\n[{arr.min():{fmt}}, {arr.max():{fmt}}]",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            if row_idx == 0:
                ax.set_title(f"{src} {ch_name}", fontsize=10)

    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Log-Intensity image grid
# ---------------------------------------------------------------------------


def plot_logI_grid(
    rcmc_dm: torch.Tensor,
    slc_dm: torch.Tensor,
    rcmc_manual: torch.Tensor,
    output_path: Path,
    clip_factor: float = 3.0,
    az_buffer: int = 0,
) -> None:
    """3-row × n-col log-intensity image grid.

    Rows:
      0. RCMC — DataModule output (what training uses)
      1. RCMC — Manual pipeline  (expected equivalent; must match row 0)
      2. SLC  — DataModule output (target ground truth)

    Convention: log-intensity ``logI = ln(re² + im²  + ε)`` for display only.
    All quality metrics must be computed on linear amplitude, not on logI.
    """
    B = rcmc_dm.shape[0]

    def _prep(tensor: torch.Tensor, vmin_phys: float, vmax_phys: float) -> np.ndarray:
        phys = minmax_inverse(tensor, vmin_phys, vmax_phys)  # (B,2,H,W) physical
        if az_buffer > 0:
            H = phys.shape[2]
            Az_core = H - 2 * az_buffer
            phys = phys[:, :, az_buffer : az_buffer + Az_core, :]
        return phys_to_logI_torch(phys).numpy()  # (B, H, W)

    row_data = [
        (_prep(rcmc_dm, RC_MIN, RC_MAX), "RCMC — DataModule"),
        (_prep(rcmc_manual, RC_MIN, RC_MAX), "RCMC — Manual"),
        (_prep(slc_dm, GT_MIN, GT_MAX), "SLC  — DataModule"),
    ]

    fig, axes = plt.subplots(3, B, figsize=(4 * B, 9), constrained_layout=True)
    if B == 1:
        axes = axes[:, np.newaxis]  # ensure 2-D indexing works

    for row_idx, (imgs, row_title) in enumerate(row_data):
        for col_idx in range(B):
            img = clip_mean_std_numpy(imgs[col_idx], factor=clip_factor)
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap="viridis", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=9)
            if row_idx == 0:
                ax.set_title(f"Patch {col_idx}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle(
        "Log-Intensity images  (logI = ln(re² + im² + ε))"
        "  —  display only, metrics on linear amplitude",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_az = args.patch_size + 2 * args.buffer
    print("\n── MAYA4 data visualisation ────────────────────────────────────────")
    print(f"  data_dir     : {args.data_dir}")
    print(f"  parts        : {args.parts}")
    print(f"  patch_size   : {args.patch_size}  buffer: {args.buffer}  → total_az: {total_az}")
    print(f"  n_patches    : {args.n_patches}   max_products: {args.max_products}")
    print(f"  output_dir   : {out_dir}")

    # ── 1. DataModule path ────────────────────────────────────────────
    print("\n[1/3] Loading via MAYA4DataModule (training path)...")
    rcmc_dm, slc_dm, coords = load_normalised_batch(args)
    B = rcmc_dm.shape[0]
    print(f"  Loaded {B} patches — RCMC {tuple(rcmc_dm.shape)}, SLC {tuple(slc_dm.shape)}")
    print_images_statistics(
        _batch_to_channel_dict(rcmc_dm, slc_dm, "DM"),
        title="DataModule output (normalised [0,1])",
    )

    # ── 2. Manual pipeline path ───────────────────────────────────────
    print("\n[2/3] Loading via manual pipeline (zarr → split → normalize)...")
    rcmc_manual, slc_manual = load_manual_pipeline_batch(coords, total_az, args.patch_size)
    print(f"  Loaded {rcmc_manual.shape[0]} patches")
    print_images_statistics(
        _batch_to_channel_dict(rcmc_manual, slc_manual, "Man"),
        title="Manual pipeline output (normalised [0,1]) — must match DataModule",
    )

    delta_rcmc = (rcmc_dm - rcmc_manual).abs()
    delta_slc = (slc_dm - slc_manual).abs()
    print(
        f"\n  Δ(DataModule − Manual):"
        f"  RCMC max={delta_rcmc.max():.2e}, mean={delta_rcmc.mean():.2e}"
        f"  |  SLC max={delta_slc.max():.2e}, mean={delta_slc.mean():.2e}"
    )
    if delta_rcmc.max() > 1e-4 or delta_slc.max() > 1e-4:
        print("  ⚠  Non-trivial discrepancy — DataModule does NOT match manual pipeline.")
    else:
        print("  ✓  DataModule and manual pipeline agree (max Δ < 1e-4).")

    # ── 3. Raw physical path ──────────────────────────────────────────
    print("\n[3/3] Loading raw physical patches (no normalization)...")
    rcmc_raw, slc_raw = load_raw_zarr_batch(coords, total_az, args.patch_size)
    print(f"  Loaded {rcmc_raw.shape[0]} patches")
    print_images_statistics(
        _batch_to_channel_dict(rcmc_raw, slc_raw, "Raw"),
        title="Raw physical values (unnormalized)",
    )

    # ── 4. Produce figures ────────────────────────────────────────────
    print("\nProducing figures...")

    plot_channel_distributions(
        rcmc_dm,
        slc_dm,
        rcmc_manual,
        slc_manual,
        rcmc_raw,
        slc_raw,
        output_path=out_dir / "channel_distributions.png",
    )

    plot_logI_grid(
        rcmc_dm,
        slc_dm,
        rcmc_manual,
        output_path=out_dir / "logI_images.png",
        clip_factor=args.clip_factor,
        az_buffer=args.buffer,
    )

    print(f"\nDone. All figures saved to {out_dir}/")
    gc.collect()


if __name__ == "__main__":
    main()
