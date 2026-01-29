import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.interpolate import interp1d

# Add path to finding src assuming running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sarpyx.processor.algorithms.constants import (
    F_REF,
    RANGE_DECIMATION_MAP,
    SPEED_OF_LIGHT_MPS,
    TX_WAVELENGTH_M,
)
from sarpyx.processor.core.constants import (
    WGS84_SEMI_MAJOR_AXIS_M,
    WGS84_SEMI_MINOR_AXIS_M,
)
from sarpyx.utils.zarr_utils import ProductHandler

from src.utils import (
    load_partitioned_dataframe,
    log_info,
    log_step,
    save_image_subsampled,
    scan_available_data_extent,
)


# ---------------------------------------------------------
# Main Processing
# ---------------------------------------------------------
def main():
    """Complete processing pipeline for manual azimuth compression."""
    t_start_total = time.time()

    PLOT_FORMAT = "png"
    VIS_DIR = Path("logs/visualizations")
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    log_step("Setup Paths")

    parser = argparse.ArgumentParser(description="Manual AZ Compression Script")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/test_complete_download/PT4/s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr",
        help="Path to the input Zarr file",
    )
    args = parser.parse_args()

    filepath = Path(args.input_file)
    if not filepath.exists() or not filepath.is_dir():
        log_info("Error: Could not find the Zarr file.")
        sys.exit(1)
    log_info(f"Loading product from: {filepath}")
    p = ProductHandler(filepath)

    # ---------------------------------------------------------
    log_step("Global Overview")

    rcmc_array = p.get_array("rcmc")
    az_array = p.get_array("az")
    log_info(f"Full Array Shape: {rcmc_array.shape}")

    save_image_subsampled(
        VIS_DIR / f"global_overview_rcmc.{PLOT_FORMAT}",
        np.abs(rcmc_array),
        title=f"Global RCMC: {filepath.name}",
    )
    save_image_subsampled(
        VIS_DIR / f"global_overview_az.{PLOT_FORMAT}",
        np.abs(az_array),
        title=f"Global AZ: {filepath.name}",
    )

    # ---------------------------------------------------------
    log_step("Metadata & Ephemeris")

    metadata = load_partitioned_dataframe(p, "metadata")
    log_info(f"Metadata shape: {metadata.shape}")
    ephemeris = load_partitioned_dataframe(p, "ephemeris")
    log_info(f"Ephemeris shape: {ephemeris.shape}")

    # ---------------------------------------------------------
    log_step("Data Loading")

    # OLD: scan_available_data_extent returned the first chunk row (0-4096) and full range.
    # We want to process the FULL Azimuth extent for correct focusing, but limited Range to save memory.

    # Define ROI for patch (global coordinates)
    roi_az_start, roi_az_end = 1000, 3000
    roi_rg_start, roi_rg_end = 10000, 12000

    # Load a strip in Range that covers our ROI (plus some margin if needed, though RCMC is range-independent here)
    # Full Azimuth (0 to end), Subset of Range
    full_az_size = rcmc_array.shape[0]
    range_margin = 0

    az_slice = slice(0, full_az_size)
    range_slice = slice(roi_rg_start - range_margin, roi_rg_end + range_margin)

    log_info("Manual Slicing for Full Azimuth Processing:")
    log_info(f"  Azimuth: {az_slice} (Size: {full_az_size})")
    log_info(f"  Range:   {range_slice} (Size: {range_slice.stop - range_slice.start})")

    # Load 2D slice
    rcmc_data = p.get_array("rcmc")[az_slice, range_slice]
    ground_truth = p.get_array("az")[az_slice, range_slice]

    log_info(f"RCMC Data shape: {rcmc_data.shape}, dtype: {rcmc_data.dtype}")
    log_info(f"Ground Truth shape: {ground_truth.shape}, dtype: {ground_truth.dtype}")

    # Check for empty data
    rcmc_abs = np.abs(rcmc_data)
    gt_abs = np.abs(ground_truth)

    log_info(
        f"RCMC Stats - Mean: {np.mean(rcmc_abs):.4f}, Max: {np.max(rcmc_abs):.4f}, Non-zeros: {np.count_nonzero(rcmc_data)}, zeros: {rcmc_data.size - np.count_nonzero(rcmc_data)}"
    )
    log_info(
        f"GT Stats   - Mean: {np.mean(gt_abs):.4f}, Max: {np.max(gt_abs):.4f}, Non-zeros: {np.count_nonzero(ground_truth)}, zeros: {ground_truth.size - np.count_nonzero(ground_truth)}"
    )

    if np.all(rcmc_data == 0):
        log_info("WARNING: Loaded data is all zeros despite chunk detection.")

    # ---------------------------------------------------------
    log_step("Initial Visualization")

    t_vis_start = time.time()

    save_image_subsampled(
        VIS_DIR / f"input_rcmc.{PLOT_FORMAT}", np.abs(rcmc_data), title="Input RCMC Crop"
    )
    save_image_subsampled(
        VIS_DIR / f"ground_truth_az.{PLOT_FORMAT}",
        np.abs(ground_truth),
        title="Ground Truth AZ Crop",
    )

    # Patch Visualization (Full Resolution)
    # Coordinates in the LOADED array
    # Since we loaded range starting at roi_rg_start, the col index 0 corresponds to roi_rg_start
    # Azimuth loaded from 0, so index matches global

    loaded_patch_rows = slice(roi_az_start, roi_az_end)
    loaded_patch_cols = slice(
        0, roi_rg_end - roi_rg_start
    )  # Full width of loaded strip corresponds to the patch width

    if rcmc_data.shape[0] >= roi_az_end:
        log_info(
            f"Saving Full Res Patches for region global rows={roi_az_start}:{roi_az_end}, global cols={roi_rg_start}:{roi_rg_end}"
        )

        # RCMC
        patch_rcmc = np.abs(rcmc_data[loaded_patch_rows, loaded_patch_cols])
        save_image_subsampled(
            VIS_DIR
            / f"patch_rcmc_{roi_az_start}_{roi_az_end}_{roi_rg_start}_{roi_rg_end}.{PLOT_FORMAT}",
            patch_rcmc,
            title=f"RCMC Patch [{roi_az_start}:{roi_az_end}, {roi_rg_start}:{roi_rg_end}]",
            target_size=max(patch_rcmc.shape),  # Force step=1
        )

        # AZ (Ground Truth)
        patch_az = np.abs(ground_truth[loaded_patch_rows, loaded_patch_cols])
        save_image_subsampled(
            VIS_DIR
            / f"patch_az_{roi_az_start}_{roi_az_end}_{roi_rg_start}_{roi_rg_end}.{PLOT_FORMAT}",
            patch_az,
            title=f"GT AZ Patch [{roi_az_start}:{roi_az_end}, {roi_rg_start}:{roi_rg_end}]",
            target_size=max(patch_az.shape),  # Force step=1
        )
    else:
        log_info("Skipping patch visualization: Data dimensions too small.")

    t_vis_end = time.time()
    log_info(f"Visualization Time: {t_vis_end - t_vis_start:.3f} s")

    # ---------------------------------------------------------
    log_step("Parameters Extraction")

    meta_row = metadata.iloc[0]

    PRI = float(
        meta_row.get("PRI", meta_row.get("pri"))
    )  # (Pulse Repetition Interval): Time between consecutive pulses
    rank = float(
        meta_row.get("Rank", meta_row.get("rank"))
    )  # Integer ambiguity number (number of pulses in flight).
    RGDEC = int(
        meta_row.get("Range Decimation", meta_row.get("range_decimation"))
    )  # (Range Decimation): Factor reducing sampling rate from the master clock.

    # Strict check for 'swst' as requested
    if "swst" in meta_row:
        SWST = float(meta_row["swst"])
    elif "SWST" in meta_row:
        SWST = float(meta_row["SWST"])
    else:
        raise KeyError("Strict metadata check failed: 'swst' key not found.")

    c = SPEED_OF_LIGHT_MPS
    wavelength = TX_WAVELENGTH_M  # carrier wavelength

    log_info(f"PRI: {PRI:.6e} s")
    log_info(f"Rank: {rank}")
    log_info(f"Range Decimation: {RGDEC}")
    log_info(f"SWST: {SWST} s")

    # Sample rates
    range_sample_freq = float(RANGE_DECIMATION_MAP[RGDEC])
    range_sample_period = 1.0 / range_sample_freq  # Sampling rate in range
    az_sample_freq = 1 / PRI
    az_sample_period = PRI

    # ---------------------------------------------------------

    log_step("Geometry Vectors")
    num_az_lines = rcmc_data.shape[0]
    num_range_samples = rcmc_data.shape[1]

    # Azimuth Frequency
    az_freq_vals = np.fft.fftfreq(num_az_lines, d=PRI)
    az_freq_vals_centered = np.fft.fftshift(az_freq_vals)

    # Slant Range
    suppressed_data_time = 320 / (8 * F_REF)
    range_start_time = SWST + suppressed_data_time

    # Correct Fast Time Vector for Range Offset
    range_offset = range_slice.start
    fast_time_vec = range_start_time + (
        range_sample_period * (np.arange(num_range_samples) + range_offset)
    )
    slant_range_vec = (
        ((rank * PRI) + fast_time_vec) * c / 2
    )  # The distance of closest approach (R0) for each range bin.

    log_info(f"Slant Range: {slant_range_vec[0]:.2f}m to {slant_range_vec[-1]:.2f}m")

    # ---------------------------------------------------------

    log_step("Effective Velocity (Vr) & D Parameter")

    if ephemeris is None or ephemeris.empty:
        log_info("Error: Ephemeris missing.")
        sys.exit(1)

    ephemeris = ephemeris.drop_duplicates(subset=["time_stamp"]).sort_values("time_stamp")

    # --- PREVIOUS IMPLEMENTATION (Optimized Vector Interpolation) ---
    # t_eph = ephemeris["time_stamp"].values.astype(float) / (2**24)
    # pos_data = ephemeris[['x', 'y', 'z']].values
    # vel_data = ephemeris[['vx', 'vy', 'vz']].values
    # interp_pos = interp1d(t_eph, pos_data, axis=0, kind='cubic', fill_value="extrapolate")
    # interp_vel = interp1d(t_eph, vel_data, axis=0, kind='cubic', fill_value="extrapolate")
    # slice_metadata = metadata.iloc[az_slice].reset_index(drop=True)
    # az_times = (slice_metadata["coarse_time"] + slice_metadata["fine_time"]).values.astype(float)
    # sat_pos = interp_pos(az_times)
    # sat_vel = interp_vel(az_times)
    # Rs = np.linalg.norm(sat_pos, axis=1)
    # Vs = np.linalg.norm(sat_vel, axis=1) # Norm after interpolation
    # ...
    # ---------------------------------------------------------------

    # --- NOTEBOOK-LIKE IMPLEMENTATION ---
    # 1. Calculate Norms/Scalars from Ephemeris FIRST (as in Notebook)
    t_eph = ephemeris["time_stamp"].values.astype(float) / (2**24)

    # Calculate scalar velocities (ECEF magnitude) for each ephemeris point
    # Ensure we use numpy values for computation
    ecef_vels = np.sqrt(
        ephemeris["vx"].values ** 2 + ephemeris["vy"].values ** 2 + ephemeris["vz"].values ** 2
    )

    # Create Interpolators
    # Note: Notebook interpolates Scalar Velocity, and Vector Position
    velocity_interp = interp1d(t_eph, ecef_vels, kind="linear", fill_value="extrapolate")

    x_interp = interp1d(t_eph, ephemeris["x"].values, kind="linear", fill_value="extrapolate")
    y_interp = interp1d(t_eph, ephemeris["y"].values, kind="linear", fill_value="extrapolate")
    z_interp = interp1d(t_eph, ephemeris["z"].values, kind="linear", fill_value="extrapolate")

    # 2. Interpolate for Azimuth Lines
    slice_metadata = metadata.iloc[az_slice].reset_index(drop=True)
    az_times = (slice_metadata["coarse_time"] + slice_metadata["fine_time"]).values.astype(float)
    log_info(f"Azimuth Time Range (Slice): {az_times[0]:.4f} to {az_times[-1]:.4f}")

    space_velocities_mps = velocity_interp(az_times)  # Vs

    x_positions = x_interp(az_times)
    y_positions = y_interp(az_times)
    z_positions = z_interp(az_times)
    position_array = np.stack((x_positions, y_positions, z_positions), axis=1)

    satellite_distance_from_center_m = np.linalg.norm(position_array, axis=1)  # Rs

    # 3. Geometry Calculations (Notebook Style)
    satellite_angular_velocity_rps = space_velocities_mps / satellite_distance_from_center_m
    satellite_latitude_rad = np.arctan2(
        position_array[:, 2], np.linalg.norm(position_array[:, :2], axis=1)
    )

    local_earth_rad_m = np.sqrt(
        (
            np.square(WGS84_SEMI_MAJOR_AXIS_M**2 * np.cos(satellite_latitude_rad))
            + np.square(WGS84_SEMI_MINOR_AXIS_M**2 * np.sin(satellite_latitude_rad))
        )
        / (
            np.square(WGS84_SEMI_MAJOR_AXIS_M * np.cos(satellite_latitude_rad))
            + np.square(WGS84_SEMI_MINOR_AXIS_M * np.sin(satellite_latitude_rad))
        )
    )

    # Broadcast to 2D
    # Dimensions: (Azimuth, Range)
    local_earth_rad_m = local_earth_rad_m[:, np.newaxis]
    satellite_distance_from_center_m = satellite_distance_from_center_m[:, np.newaxis]
    slant_range_vec_m = slant_range_vec[np.newaxis, :]  # R0

    cos_beta = (
        np.square(local_earth_rad_m)
        + np.square(satellite_distance_from_center_m)
        - np.square(slant_range_vec_m)
    ) / (2 * local_earth_rad_m * satellite_distance_from_center_m)

    ground_velocities_mps = (
        local_earth_rad_m * satellite_angular_velocity_rps[:, np.newaxis] * cos_beta
    )

    # Effective Velocity Vr
    effective_velocities_mps = np.sqrt(space_velocities_mps[:, np.newaxis] * ground_velocities_mps)

    # For D calculation
    Vr = effective_velocities_mps

    log_info(f"Vr Mean: {np.mean(Vr):.2f} m/s")

    # ---------------------------------------------------------

    log_step("Azimuth Compression")
    t_comp_start = time.time()

    # FFT (Result is in Range-Doppler domain)
    rcmc_fft = fftshift(fft(rcmc_data, axis=0), axes=0)

    # --- Exact Matched Filter (Notebook Style) ---
    # D(f_eta, Vr) = sqrt(1 - (lambda * f_eta)^2 / (4 * Vr^2))
    f_az_2d = az_freq_vals_centered[:, np.newaxis]

    # Argument inside sqrt
    # Note: epsilon added to avoid division by zero or negative in sqrt (though physically shouldn't happen for valid beam)
    # Using complex sqrt or clipping could be safer, but assuming valid data:
    D_arg = 1 - (wavelength**2 * f_az_2d**2) / (4 * Vr**2)
    D = np.sqrt(
        D_arg.astype(np.complex64)
    )  # Use complex cast to handle potential small negatives gracefully

    # Filter Formula: exp( 4j * pi * R0 * D / lambda )
    # This includes the phase constant and the correct sign convention used in the notebook
    match_filter = np.exp(4.0j * np.pi * slant_range_vec_m * D / wavelength)

    # Apply
    compressed_fft = rcmc_fft * match_filter

    # IFFT
    compressed_data = ifft(ifftshift(compressed_fft, axes=0), axis=0)

    t_comp_end = time.time()

    log_info(f"Compression Time: {t_comp_end - t_comp_start:.3f} s")

    log_step("Saving Results")
    save_image_subsampled(
        VIS_DIR / f"output_compressed.{PLOT_FORMAT}",
        np.abs(compressed_data),
        title="Output Compressed AZ",
    )

    if compressed_data.shape[0] >= roi_az_end:
        # Patch Visualization for Compressed Data
        # Using same slice indices as above
        patch_compressed = np.abs(compressed_data[loaded_patch_rows, loaded_patch_cols])
        save_image_subsampled(
            VIS_DIR
            / f"patch_output_compressed_{roi_az_start}_{roi_az_end}_{roi_rg_start}_{roi_rg_end}.{PLOT_FORMAT}",
            patch_compressed,
            title=f"Output Compressed Patch [{roi_az_start}:{roi_az_end}, {roi_rg_start}:{roi_rg_end}]",
            target_size=max(patch_compressed.shape),
        )

        log_step("Saving Comparison Overlay")
        # Global Subsampled Comparison
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Subsample for display (e.g. step 50 for large image)
        step = 50

        axs[0].imshow(
            np.abs(rcmc_data)[::step, ::step], aspect="auto", cmap="gray", interpolation="none"
        )
        axs[0].set_title("Input RCMC (Sampled)")

        axs[1].imshow(
            np.abs(ground_truth)[::step, ::step], aspect="auto", cmap="gray", interpolation="none"
        )
        axs[1].set_title("Ground Truth Azimuth (Sampled)")

        axs[2].imshow(
            np.abs(compressed_data)[::step, ::step],
            aspect="auto",
            cmap="gray",
            interpolation="none",
        )
        axs[2].set_title("Output Compressed (Sampled)")

        plt.tight_layout()
        plt.savefig(VIS_DIR / "comparison_overview.png")
        plt.close()

    # ---------------------------------------------------------

    log_step("correlation Analysis")

    # Debug: Print Metadata Columns to check for Doppler
    # log_info(f"Metadata Columns: {list(metadata.columns)}")

    # Crop Center (Margin 1000 pixels)
    # We are now slicing in Range, but Azimuth is full.
    # We should perform correlation on the ROI we visualized or the full loaded strip?
    # Let's use the full loaded strip (subset of range) with some margin in Azimuth to avoid edges

    MARGIN_AZ = 1000

    if num_az_lines > (2 * MARGIN_AZ):
        log_info(f"Cropping {MARGIN_AZ} pixels from top/bottom Azimuth for correlation.")
        slit = slice(MARGIN_AZ, -MARGIN_AZ)

        comp_crop = compressed_data[slit, :]
        gt_crop = ground_truth[slit, :]

        # Also crop magnitude for testing
        mag_compressed = np.abs(compressed_data)
        mag_gt = np.abs(ground_truth)

        mag_comp_crop = mag_compressed[slit, :]
        mag_gt_crop = mag_gt[slit, :]
    else:
        log_info(
            f"Warning: Chunk size ({num_az_lines}) too small for {MARGIN_AZ} pixel margin. Using full slice."
        )
        comp_crop = compressed_data
        gt_crop = ground_truth
        mag_compressed = np.abs(compressed_data)
        mag_gt = np.abs(ground_truth)
        mag_comp_crop = mag_compressed
        mag_gt_crop = mag_gt

    log_info(f"Correlation Slice Shape: {comp_crop.shape}")

    # Helper for Correlation
    def get_corr(a, b):
        return np.abs(np.sum(a * np.conj(b))) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_mag_corr(a, b):
        # Pearson correlation for magnitude
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.corrcoef(a_flat, b_flat)[0, 1]

    # 1. Standard
    corr_c = get_corr(comp_crop, gt_crop)
    corr_m = get_mag_corr(mag_comp_crop, mag_gt_crop)
    log_info(f"Standard Correlation - Complex: {corr_c:.4f}, Magnitude: {corr_m:.4f}")

    # 2. Flipped Azimuth
    comp_flip = np.flipud(comp_crop)
    corr_c_flip = get_corr(comp_flip, gt_crop)
    corr_m_flip = get_mag_corr(np.flipud(mag_comp_crop), mag_gt_crop)
    log_info(f"Flipped Azimuth    - Complex: {corr_c_flip:.4f}, Magnitude: {corr_m_flip:.4f}")

    # 3. Conjugate (Standard)
    # Already computed in Standard (a * conj(b))

    # 4. Raw Complex Product Sum (without conj?)
    # Sometimes people define correlation differently

    # 5. Check Center Line Shift (Cross Correlation)
    mid_col = comp_crop.shape[1] // 2
    col_comp = mag_comp_crop[:, mid_col]
    col_gt = mag_gt_crop[:, mid_col]

    # Simple peak search using argmax of convolution
    # Normalize
    col_comp = (col_comp - np.mean(col_comp)) / (np.std(col_comp) + 1e-9)
    col_gt = (col_gt - np.mean(col_gt)) / (np.std(col_gt) + 1e-9)

    xcorr = np.correlate(col_gt, col_comp, mode="full")
    lag = np.argmax(xcorr) - (len(col_comp) - 1)
    peak = np.max(xcorr) / len(col_comp)
    log_info(f"1D X-Corr Peak at Lag: {lag} pixels (Val: {peak:.4f})")

    if abs(lag) > 0 and abs(lag) < 500:
        log_info(f"Trying shifted correlation with lag {lag}...")
        # Shift comp_crop by lag
        if lag > 0:
            # GT is ahead of Comp. Shift Comp down?
            # lag = argmax(GT * Comp_rev). Positive lag means GT is shifted vs Comp.
            # We need to roll Comp?
            pass  # Too complex to implement robust shift-and-crop in one go, but lag tells us alignment.

    corr_m = np.corrcoef(mag_comp_crop.flatten(), mag_gt_crop.flatten())[0, 1]

    log_info(f"Result Complex Correlation (Center): {corr_c:.4f}")
    log_info(f"Result Magnitude Correlation (Center): {corr_m:.4f}")

    t_end_total = time.time()
    log_step("Summary")
    log_info(f"Total Execution Time: {t_end_total - t_start_total:.3f} s")


if __name__ == "__main__":
    main()
