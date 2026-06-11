"""Study the effect of the azimuth-focusing buffer size.

Standalone experiment (not part of the training pipeline): focuses a fixed RCMC
block with the sarpyx ``CoarseRDA`` processor under different azimuth buffer
sizes and inspects how much azimuth context the focusing needs before the
result stabilises. Used to justify the ``azimuth_buffer`` choice in the configs.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add path to finding srp/sarpyx if needed, assuming running from root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sarpyx.processor.algorithms.constants import (
    RANGE_DECIMATION_MAP,
    TX_WAVELENGTH_M,
)
from sarpyx.processor.core.focus import CoarseRDA

# Sarpyx imports
from sarpyx.utils.zarr_utils import ProductHandler

import src.utils as utils
from src.utils import compute_correlation, compute_mag_correlation, log_info, log_step


def process_block(p, az_slice, rg_slice, full_az_size, verbose=True):
    """Process a specific block of data defined by az_slice and rg_slice.

    Returns:
        focused_data: np.ndarray (complex)
        params: dict of physics parameters (from the processor)
    """

    # 1. Load Data
    roi_rcmc = p.get_array("rcmc")[az_slice, rg_slice]

    # 2. Adapting Metadata
    metadata_full = utils.load_partitioned_dataframe(p, "metadata")
    ephemeris_full = utils.load_partitioned_dataframe(p, "ephemeris")

    metadata_slice = metadata_full.copy()

    # Azimuth Slicing
    if az_slice.start > 0 or az_slice.stop < full_az_size:
        metadata_slice = metadata_slice.iloc[az_slice].reset_index(drop=True)

    # Range Slicing (SWST update)
    if rg_slice.start > 0:
        meta_row = metadata_slice.iloc[0]
        rgdec = meta_row.get("Range Decimation", meta_row.get("range_decimation"))
        if rgdec is None:
            print("Warning: Range Decimation not found, using default 9 (likely)")
            rgdec = 9

        range_freq = float(RANGE_DECIMATION_MAP[int(rgdec)])
        time_offset = rg_slice.start / range_freq
        swst_col = "swst"
        metadata_slice[swst_col] = metadata_slice[swst_col] + time_offset

    # 3. Initialize Processor
    mock_raw_data = {"echo": roi_rcmc, "metadata": metadata_slice, "ephemeris": ephemeris_full}

    # Mute verbosity usually
    processor = CoarseRDA(mock_raw_data, verbose=verbose, memory_efficient=False)
    del roi_rcmc, mock_raw_data
    gc.collect()

    # 4. Processing
    # FFT (No Shift)
    processor.radar_data = np.fft.fft(processor.radar_data, axis=0)

    processor._compute_effective_velocities()
    processor.wavelength = TX_WAVELENGTH_M

    processor.az_freq_vals = np.arange(
        -processor.az_sample_freq / 2,
        processor.az_sample_freq / 2,
        processor.az_sample_freq / processor.len_az_line,
    )
    if len(processor.az_freq_vals) != processor.len_az_line:
        processor.az_freq_vals = np.linspace(
            -processor.az_sample_freq / 2,
            processor.az_sample_freq / 2,
            processor.len_az_line,
            endpoint=False,
        )

    mean_effective_velocities = np.mean(processor.effective_velocities, axis=1)
    processor.D = np.sqrt(
        1
        - (processor.wavelength**2 * processor.az_freq_vals**2)
        / (4 * mean_effective_velocities**2)
    )

    processor._perform_azimuth_compression_efficient()

    # Extract params for calculation
    params = {
        "wavelength": processor.wavelength,
        "slant_range_vec": processor.slant_range_vec,
        "effective_velocities": processor.effective_velocities,
        "az_sample_freq": processor.az_sample_freq,
    }

    return processor.radar_data, params


def main():
    """Main entry point."""
    # Setup
    input_file = Path(
        "data/TEST/FOCUSED_v4/s1a-s6-raw-s-vh-20240502t195132-20240502t195153-053697-0685cc.zarr/"
    )
    if not input_file.exists():
        # Fallback
        input_file = Path(
            "data/test_complete_download/PT4/s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr"
        )

    VIS_DIR = Path("logs/local_az_compr")
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    log_info(f"Using file: {input_file}")
    p = ProductHandler(input_file)

    full_rcmc_shape = p.get_array("rcmc").shape
    full_az_size = full_rcmc_shape[0]

    # Target Region (The "Patch" we want to recover correctly)
    target_az_start = 7000
    target_az_len = 2000
    target_az_end = target_az_start + target_az_len

    target_rg_start = 5000
    target_rg_end = 7000
    target_rg_slice = slice(target_rg_start, target_rg_end)

    # 1. Run "Reference" (Large Context)
    REF_MARGIN = 3000
    ref_az_start = max(0, target_az_start - REF_MARGIN)
    ref_az_end = min(full_az_size, target_az_end + REF_MARGIN)
    ref_az_slice = slice(ref_az_start, ref_az_end)

    log_step(f"Running Reference Processing (Context: {ref_az_slice})")
    ref_result, params = process_block(
        p, ref_az_slice, target_rg_slice, full_az_size, verbose=True
    )

    # Get only the target area from reference
    ref_target = np.abs(
        ref_result[target_az_start - ref_az_start : target_az_end - ref_az_start, :]
    )

    # Save Reference Image
    utils.save_image_subsampled(
        VIS_DIR / "reference_block.png",
        ref_target,
        title="Reference Block (Full Context)",
        target_size=ref_target.shape[0],
    )

    # Extract Target from Reference
    offset = target_az_start - ref_az_start
    ref_target = ref_result[offset : offset + target_az_len, :]

    # ---------------------------------------------------------
    # Theoretical Calculation
    # ---------------------------------------------------------
    log_step("Theoretical Buffer Estimation")

    # Extract params
    R0 = np.max(params["slant_range_vec"])  # Slant Range (m)
    log_info(
        f"Slant Range (R0) - Min: {np.min(params['slant_range_vec']):.2f} m | Mean: {np.mean(params['slant_range_vec']):.2f} m | Max: {R0:.2f} m"
    )
    wavelength = params["wavelength"]  # Wavelength (m)
    L_ant = 12.3  # Antenna Length (m) - Sentinel-1 Standard

    # Calculate Delta x_az
    # V_g approx V_eff for this estimation
    V_eff = np.mean(params["effective_velocities"])
    log_info(
        f"Effective Velocity (V_eff) - Min: {np.min(params['effective_velocities']):.2f} m/s | Mean: {V_eff:.2f} m/s | Max: {np.max(params['effective_velocities']):.2f} m/s"
    )
    PRF = params["az_sample_freq"]
    delta_x_az = V_eff / PRF

    # Formula: N_az = (R0 * lambda) / (L_ant * delta_x_az)
    N_az_theoretical = (R0 * wavelength) / (L_ant * delta_x_az)

    log_info("Parameters:")
    log_info(f"  -> Slant Range (R0): {R0:.2f} m")
    log_info(f"  -> Wavelength (mu): {wavelength:.5f} m")
    log_info(f"  -> Antenna Length (L_ant): {L_ant} m (Assumed)")
    log_info(f"  -> Eff. Velocity (V): {V_eff:.2f} m/s")
    log_info(f"  -> PRF: {PRF:.2f} Hz")
    log_info(f"  -> Azimuth Resolution (delta_x_az): {delta_x_az:.4f} m")

    log_info("Calculated Theoretical Buffer Size (N_az):")
    log_info(f"  -> {N_az_theoretical:.2f} cells (Total Aperture Length)")
    log_info(f"  -> {N_az_theoretical/2:.2f} cells (Required Margin/Truncation per side)")

    # ---------------------------------------------------------
    # Experiments
    # ---------------------------------------------------------
    margins = [0, 100, 250, 500, 1000, 1500, 2000]
    errors = []
    correlations_c = []
    correlations_m = []

    log_step("Running Margin Experiments")

    for margin in margins:
        m_az_start = max(0, target_az_start - margin)
        m_az_end = min(full_az_size, target_az_end + margin)
        m_az_slice = slice(m_az_start, m_az_end)

        log_info(f"Testing Margin: {margin} (Context: {m_az_slice})")

        try:
            m_result, _ = process_block(
                p, m_az_slice, target_rg_slice, full_az_size, verbose=False
            )

            # Extract Target
            offset_m = target_az_start - m_az_start
            m_target = m_result[offset_m : offset_m + target_az_len, :]

            # 1. Error (MAE)
            diff = np.abs(m_target - ref_target)
            error = np.mean(diff) / np.mean(np.abs(ref_target))
            errors.append(error)

            # 2. Correlations
            # Crop extreme edges of the target itself before correlation?
            # Ideally m_target should match ref_target perfectly if margin is enough.
            cc = compute_correlation(m_target, ref_target)
            cm = compute_mag_correlation(m_target, ref_target)
            correlations_c.append(cc)
            correlations_m.append(cm)

            log_info(
                f"  -> Rel. Error: {error:.6f} | Corr(Complex): {cc:.6f} | Corr(Mag): {cm:.6f}"
            )

            # Save Visualization
            if margin in [0, 500, 1000, 2000]:  # Save selected interesting ones
                utils.save_image_subsampled(
                    VIS_DIR / f"block_margin_{margin}.png",
                    np.abs(m_target),
                    title=f"Result with Margin {margin} (Err: {error:.2%})",
                    target_size=m_target.shape[0],
                )

        except Exception as e:
            log_info(f"  -> Failed: {e}")
            errors.append(np.nan)
            correlations_c.append(np.nan)
            correlations_m.append(np.nan)

    # Plot results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(margins, errors, "o-", label="Relative MAE")
    # Plot theoretic threshold
    plt.axvline(
        x=N_az_theoretical / 2,
        color="r",
        linestyle="--",
        label=f"Theoretical Margin (~{int(N_az_theoretical/2)})",
    )
    plt.xlabel("Azimuth Margin (pixels)")
    plt.ylabel("Relative MAE")
    plt.title("Focusing Error vs Margin")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")

    plt.subplot(2, 1, 2)
    plt.plot(margins, correlations_c, "s-", label="Complex Correlation")
    plt.plot(margins, correlations_m, "^-", label="Magnitude Correlation")
    plt.axvline(x=N_az_theoretical / 2, color="r", linestyle="--", label="Theoretical Margin")
    plt.xlabel("Azimuth Margin (pixels)")
    plt.ylabel("Correlation")
    plt.ylim(0.8, 1.005)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(VIS_DIR / "az_margin_analysis.png")
    log_info(f"Saved analysis plot to {VIS_DIR / 'az_margin_analysis.png'}")


if __name__ == "__main__":
    main()
