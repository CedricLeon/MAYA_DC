r"""Azimuth compression pipeline validation script (F10).

Compares three focusing results on the same RCMC patch:

  1. **GT SLC**      — the ``az`` product stored in the MAYA4 zarr (ground truth).
  2. **CoarseRDA**   — the sarpyx ``CoarseRDA`` processor (reference algorithm).
  3. **Custom FFT**  — our standalone ``full_azimuth_compress_batch`` from
                       ``src/utils/sarpyx_azimuth_compression.py`` (no network).

Metrics reported for CoarseRDA vs GT and Custom vs GT:
  - Complex correlation  \|<x, y*>\| / (||x|| ||y||)
  - Magnitude correlation
  - PSNR [dB]  (data_range = max(\|GT\|))
  - SSIM       (data_range = max(\|GT\|))

Usage::

    python scripts/validate_azimuth_pipeline.py
    python scripts/validate_azimuth_pipeline.py --input_file data/PT4/sample.zarr
    python scripts/validate_azimuth_pipeline.py --patch_az 3000 --patch_rg 12000 --patch_size 1024

The script uses an azimuth buffer so that the Custom FFT path operates on the
same number of azimuth lines as CoarseRDA (full ROI), then strips the buffer
before comparing against GT — exactly replicating the training pipeline.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")  # no display needed

# ---------------------------------------------------------------------------
# Path setup — works regardless of cwd
# ---------------------------------------------------------------------------
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sarpyx.processor.algorithms.constants import RANGE_DECIMATION_MAP, TX_WAVELENGTH_M
from sarpyx.processor.core.focus import CoarseRDA
from sarpyx.utils.zarr_utils import ProductHandler

import src.utils as utils
from src.utils import log_info, log_step
from src.utils.sarpyx_azimuth_compression import (
    compute_azimuth_filter,
    full_azimuth_compress_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORR_FMT = "  Complex corr : {:.6f}"
_MAGCORR_FMT = "  Magnitude corr: {:.6f}"
_PSNR_FMT = "  PSNR          : {:.2f} dB"
_SSIM_FMT = "  SSIM          : {:.6f}"


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    r"""PSNR [dB] on magnitude, with data_range = max(\|ref\|)."""
    data_range = np.abs(ref).max()
    mse = np.mean((np.abs(pred) - np.abs(ref)) ** 2)
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def _ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    """SSIM on magnitude images (simplified, window-based via scipy)."""
    from skimage.metrics import structural_similarity as sk_ssim

    p_mag = np.abs(pred).astype(np.float64)
    r_mag = np.abs(ref).astype(np.float64)
    data_range = r_mag.max()
    result = sk_ssim(p_mag, r_mag, data_range=data_range)
    return float(result) if not isinstance(result, tuple) else float(result[0])


def _print_metrics(label: str, pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute and print metrics comparing ``pred`` against ``ref`` with a header ``label``."""
    log_info(f"\n{'─'*60}")
    log_info(f"Metrics: {label}")
    cc = utils.compute_correlation(pred, ref)
    cm = utils.compute_mag_correlation(pred, ref)
    psnr = _psnr(pred, ref)
    ssim = _ssim(pred, ref)
    log_info(_CORR_FMT.format(cc))
    log_info(_MAGCORR_FMT.format(cm))
    log_info(_PSNR_FMT.format(psnr))
    log_info(_SSIM_FMT.format(ssim))
    return {"complex_corr": cc, "mag_corr": cm, "psnr_db": psnr, "ssim": ssim}


def _run_coarserda(roi_rcmc: np.ndarray, metadata_slice, ephemeris_full) -> np.ndarray:
    """Run the CoarseRDA reference algorithm on ``roi_rcmc`` (full azimuth ROI)."""
    mock = {"echo": roi_rcmc, "metadata": metadata_slice, "ephemeris": ephemeris_full}
    proc = CoarseRDA(mock, verbose=False, memory_efficient=False)
    del mock
    gc.collect()

    proc.radar_data = np.fft.fft(proc.radar_data, axis=0)
    proc._compute_effective_velocities()
    proc.wavelength = TX_WAVELENGTH_M

    proc.az_freq_vals = np.linspace(
        -proc.az_sample_freq / 2,
        proc.az_sample_freq / 2,
        proc.len_az_line,
        endpoint=False,
    )
    mean_V = np.mean(proc.effective_velocities, axis=1)
    proc.D = np.sqrt(
        np.maximum(1.0 - (proc.wavelength**2 * proc.az_freq_vals**2) / (4.0 * mean_V**2), 0.0)
    )
    proc._perform_azimuth_compression_efficient()
    result = proc.radar_data.copy()
    del proc
    gc.collect()
    return result


def _run_custom_fft(
    roi_rcmc: np.ndarray,
    metadata_slice,
    ephemeris_full,
    buffer_size: int,
) -> np.ndarray:
    """Run our custom torch-FFT pipeline on ``roi_rcmc`` (with buffer included).

    Returns the **core** patch (buffer stripped), matching the training pipeline
    exactly:  ``full_azimuth_compress_batch`` → strip buffer → numpy.
    """
    # Convert (Az, Rg) complex128 → (1, 2, Az, Rg) float32  [real/imag channels]
    rcmc_arr = np.asarray(roi_rcmc)  # ensure numpy array (not zarr proxy)
    rcmc_t = torch.from_numpy(
        np.stack([np.real(rcmc_arr), np.imag(rcmc_arr)], axis=0)[np.newaxis]  # (1,2,Az,Rg)
    ).float()

    slc_t = full_azimuth_compress_batch(
        rcmc_t,
        metadata_batch=[metadata_slice],
        ephemeris_batch=[ephemeris_full],
        buffer_size=buffer_size,
        device="cpu",
    )  # (1, 2, Az_core, Rg)

    ri = slc_t[0].numpy()  # (2, Az_core, Rg)
    return ri[0] + 1j * ri[1]  # (Az_core, Rg) complex64→128


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main entry point."""
    t_start = time.time()

    parser = argparse.ArgumentParser(
        description="Validate custom azimuth compression vs CoarseRDA"
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--vis_dir", type=str, default="logs/visualizations")
    parser.add_argument("--patch_az", type=int, default=None, help="Patch azimuth start")
    parser.add_argument("--patch_rg", type=int, default=None, help="Patch range start")
    parser.add_argument("--patch_size", type=int, default=512, help="Core patch size (az and rg)")
    parser.add_argument(
        "--buffer",
        type=int,
        default=512,
        help="Azimuth buffer lines added on each side for the custom FFT path",
    )
    args = parser.parse_args()

    # --- File selection ---------------------------------------------------
    _DEFAULT_FILES = [
        "data/test_complete_download/PT4/s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr",
        "data/PT4/s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr",
    ]
    if args.input_file:
        filepath = Path(args.input_file)
    else:
        filepath = next((Path(f) for f in _DEFAULT_FILES if Path(f).exists()), None)
        if filepath is None:
            raise FileNotFoundError("No default zarr file found. Pass --input_file explicitly.")
    log_info(f"Input file: {filepath}")

    filename_shorthand = filepath.stem.split("-")[-1]
    VIS_DIR = Path(args.vis_dir) / f"validate_az_{filename_shorthand}"
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    p = ProductHandler(str(filepath))

    # --- Available extent (handles incomplete downloads) ------------------
    log_step("Checking available data extent")
    av_rcmc_az, av_rcmc_rg = utils.scan_available_data_extent(filepath, "rcmc", verbose=False)
    av_az_az, av_az_rg = utils.scan_available_data_extent(filepath, "az", verbose=False)
    eff_az = min(av_rcmc_az.stop, av_az_az.stop)
    eff_rg = min(av_rcmc_rg.stop, av_az_rg.stop)
    log_info(f"Effective available extent: az={eff_az}, rg={eff_rg}")

    # --- Patch coordinates ------------------------------------------------
    _DEFAULTS = {"05d521": (3000, 12000), "05e631": (7000, 5000), "0685cc": (7000, 5000)}
    default_az, default_rg = _DEFAULTS.get(filename_shorthand, (1000, 1000))
    patch_az_start = args.patch_az if args.patch_az is not None else default_az
    patch_rg_start = args.patch_rg if args.patch_rg is not None else default_rg
    patch_size = args.patch_size
    buffer = args.buffer

    patch_az_end = min(patch_az_start + patch_size, eff_az)
    patch_rg_end = min(patch_rg_start + patch_size, eff_rg)
    patch_az_start = patch_az_end - patch_size  # re-anchor if clamped

    log_info(
        f"Core patch: az=[{patch_az_start}:{patch_az_end}], "
        f"rg=[{patch_rg_start}:{patch_rg_end}], size={patch_size}x{patch_size}"
    )
    log_info(f"Azimuth buffer: {buffer} lines each side")

    # CoarseRDA and GT use the full azimuth ROI (az=0..eff_az) to avoid edge effects.
    # Custom FFT uses the buffered patch (patch_az ± buffer) which mirrors training.
    roi_az_slice = slice(0, eff_az)
    roi_rg_slice = slice(patch_rg_start, patch_rg_end)

    buf_az_start = max(0, patch_az_start - buffer)
    buf_az_end = min(eff_az, patch_az_end + buffer)
    actual_buf_top = patch_az_start - buf_az_start  # may be < buffer at image edge
    actual_buf_bot = buf_az_end - patch_az_end
    buf_az_slice = slice(buf_az_start, buf_az_end)

    log_info(
        f"Buffered ROI for custom FFT: az=[{buf_az_start}:{buf_az_end}] "
        f"(top_buf={actual_buf_top}, bot_buf={actual_buf_bot})"
    )

    # --- Load data --------------------------------------------------------
    log_step("Loading data")
    roi_rcmc_full = np.asarray(p.get_array("rcmc")[roi_az_slice, roi_rg_slice])
    buf_rcmc = np.asarray(p.get_array("rcmc")[buf_az_slice, roi_rg_slice])
    patch_gt_slc = np.asarray(p.get_array("az")[slice(patch_az_start, patch_az_end), roi_rg_slice])
    log_info(f"Full ROI RCMC shape: {roi_rcmc_full.shape}")
    log_info(f"Buffered RCMC shape: {buf_rcmc.shape}")
    log_info(f"GT SLC patch shape : {patch_gt_slc.shape}")

    # --- Prepare metadata -------------------------------------------------
    log_step("Preparing metadata")
    metadata_full = utils.load_partitioned_dataframe(p, "metadata")
    ephemeris_full = utils.load_partitioned_dataframe(p, "ephemeris")

    def _slice_and_shift_metadata(meta_full, az_slice, rg_start):
        meta = meta_full.copy()
        if az_slice.start > 0 or az_slice.stop < eff_az:
            meta = meta.iloc[az_slice].reset_index(drop=True)
        if rg_start > 0:
            meta_row = meta.iloc[0]
            rgdec = meta_row.get("Range Decimation", meta_row.get("range_decimation"))
            range_freq = float(RANGE_DECIMATION_MAP[int(rgdec)])
            meta["swst"] = meta["swst"] + rg_start / range_freq
        return meta

    meta_full_roi = _slice_and_shift_metadata(metadata_full, roi_az_slice, patch_rg_start)
    meta_buf = _slice_and_shift_metadata(metadata_full, buf_az_slice, patch_rg_start)

    # --- Run CoarseRDA (full ROI, reference) ------------------------------
    log_step("Running CoarseRDA (reference)")
    t0 = time.time()
    coarserda_full = _run_coarserda(roi_rcmc_full, meta_full_roi, ephemeris_full)
    log_info(f"CoarseRDA done in {time.time()-t0:.1f}s, output shape: {coarserda_full.shape}")

    # Extract patch from CoarseRDA full result
    local_az = slice(patch_az_start - roi_az_slice.start, patch_az_end - roi_az_slice.start)
    coarserda_patch = coarserda_full[local_az, :]
    del roi_rcmc_full, coarserda_full
    gc.collect()

    # --- Run custom FFT (buffered patch) ----------------------------------
    log_step("Running custom torch-FFT pipeline")
    t0 = time.time()
    custom_patch = _run_custom_fft(buf_rcmc, meta_buf, ephemeris_full, actual_buf_top)
    log_info(f"Custom FFT done in {time.time()-t0:.1f}s, output shape: {custom_patch.shape}")
    del buf_rcmc
    gc.collect()

    # Custom patch may be slightly smaller than patch_size if buffer was clipped at edge.
    # Align all three arrays to the smallest common size.
    min_az = min(coarserda_patch.shape[0], custom_patch.shape[0], patch_gt_slc.shape[0])
    min_rg = min(coarserda_patch.shape[1], custom_patch.shape[1], patch_gt_slc.shape[1])
    coarserda_patch = coarserda_patch[:min_az, :min_rg]
    custom_patch = custom_patch[:min_az, :min_rg]
    patch_gt_slc = patch_gt_slc[:min_az, :min_rg]
    log_info(f"Aligned patch size: {min_az} x {min_rg}")

    # --- Metrics ----------------------------------------------------------
    log_step("Computing metrics")
    metrics_coarserda = _print_metrics("CoarseRDA vs GT SLC", coarserda_patch, patch_gt_slc)
    metrics_custom = _print_metrics("Custom FFT vs GT SLC", custom_patch, patch_gt_slc)
    metrics_cross = _print_metrics("Custom FFT vs CoarseRDA", custom_patch, coarserda_patch)

    # --- Difference images ------------------------------------------------
    diff_coarserda = np.abs(coarserda_patch) - np.abs(patch_gt_slc)
    diff_custom = np.abs(custom_patch) - np.abs(patch_gt_slc)
    diff_cross = np.abs(custom_patch) - np.abs(coarserda_patch)

    # --- Plots ------------------------------------------------------------
    log_step("Saving figures")

    # 1. Side-by-side magnitude comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmax = np.percentile(np.abs(patch_gt_slc), 98)
    imgs = [np.abs(patch_gt_slc), np.abs(coarserda_patch), np.abs(custom_patch), None]
    titles = [
        "GT SLC (az product)",
        f"CoarseRDA\ncc={metrics_coarserda['complex_corr']:.4f}  PSNR={metrics_coarserda['psnr_db']:.1f}dB",
        f"Custom FFT\ncc={metrics_custom['complex_corr']:.4f}  PSNR={metrics_custom['psnr_db']:.1f}dB",
        "Custom vs CoarseRDA diff\n(residual)",
    ]
    for ax, img, title in zip(axes[:3], imgs[:3], titles[:3]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    # Diff image (use symmetric colormap)
    vd = np.percentile(np.abs(diff_cross), 99)
    axes[3].imshow(diff_cross, cmap="RdBu_r", vmin=-vd, vmax=vd, aspect="auto")
    axes[3].set_title(titles[3], fontsize=9)
    axes[3].axis("off")
    plt.suptitle(
        f"Azimuth compression validation  |  patch az=[{patch_az_start}:{patch_az_end}]  rg=[{patch_rg_start}:{patch_rg_end}]",
        fontsize=10,
    )
    plt.tight_layout()
    out = VIS_DIR / "comparison_magnitude.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log_info(f"Saved: {out}")

    # 2. Residual magnitude error maps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vd = max(
        np.percentile(np.abs(diff_coarserda), 99),
        np.percentile(np.abs(diff_custom), 99),
        1e-12,
    )
    axes[0].imshow(diff_coarserda, cmap="RdBu_r", vmin=-vd, vmax=vd, aspect="auto")
    axes[0].set_title(f"CoarseRDA − GT  (cc={metrics_coarserda['complex_corr']:.4f})", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(diff_custom, cmap="RdBu_r", vmin=-vd, vmax=vd, aspect="auto")
    axes[1].set_title(f"Custom FFT − GT  (cc={metrics_custom['complex_corr']:.4f})", fontsize=9)
    axes[1].axis("off")
    plt.suptitle("Magnitude residual vs GT SLC", fontsize=10)
    plt.tight_layout()
    out = VIS_DIR / "residuals_vs_gt.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log_info(f"Saved: {out}")

    # --- Summary table ----------------------------------------------------
    log_step("Summary")
    header = f"{'Metric':<22} {'CoarseRDA vs GT':>18} {'Custom FFT vs GT':>18} {'Custom vs CoarseRDA':>20}"
    log_info(header)
    log_info("─" * len(header))
    for key, label in [
        ("complex_corr", "Complex corr"),
        ("mag_corr", "Magnitude corr"),
        ("psnr_db", "PSNR [dB]"),
        ("ssim", "SSIM"),
    ]:
        v1 = metrics_coarserda[key]
        v2 = metrics_custom[key]
        v3 = metrics_cross[key]
        log_info(f"{label:<22} {v1:>18.4f} {v2:>18.4f} {v3:>20.4f}")

    log_info(f"\nTotal time: {time.time() - t_start:.1f}s")
    log_info(f"Output images saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
