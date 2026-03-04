import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

# Add path to finding srp/sarpyx if needed, assuming running from root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sarpyx.processor.algorithms.constants import (  # type: ignore
    RANGE_DECIMATION_MAP,
    TX_WAVELENGTH_M,
)
from sarpyx.processor.core.focus import CoarseRDA  # type: ignore

# Sarpyx imports
from sarpyx.utils.zarr_utils import ProductHandler  # type: ignore

import src.utils as utils
from src.utils import log_info, log_step


def main():
    """Complete processing pipeline using Sarpyx for Azimuth Compression on RCMC data."""
    t_start_total = time.time()

    # --- ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Sarpyx Azimuth Compression Script")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input Zarr product (contains rcmc and az groups)",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="logs/visualizations_sarpyx",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--full_image",
        action="store_true",
        help="Process the entire image (ignores hardcoded ROI in range)",
    )
    parser.add_argument("--debug_plots", action="store_true", help="Enable extra debug plots")
    args = parser.parse_args()
    # print all args
    log_info(f"Arguments: {args}")

    # --- SETUP ---
    log_step("Setup Paths")

    if args.input_file is None:  # Default fallback
        filepath = Path(
            "data/test_complete_download/PT4/s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr"
        )
    else:
        filepath = Path(args.input_file)

    filename_shorthand = filepath.stem.split("-")[-1]
    VIS_DIR = Path(args.vis_dir) / filename_shorthand
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_FORMAT = "png"

    if not filepath.exists() or not filepath.is_dir():
        raise FileNotFoundError(f"Error: Could not find the Zarr file at {filepath}")

    log_info(f"Loading product from: {filepath}")
    p = ProductHandler(filepath)

    # ---------------------------------------------------------
    log_step("Data Loading")

    full_rcmc = p.get_array("rcmc")
    full_az_size = full_rcmc.shape[0]
    full_rg_size = full_rcmc.shape[1]

    # use scan_available_data_extent to verify the whole image is downloaded
    available_RCMC_az_slice, available_RCMC_rg_slice = utils.scan_available_data_extent(
        filepath, array_name="rcmc", verbose=False
    )
    available_AZ_az_slice, available_AZ_rg_slice = utils.scan_available_data_extent(
        filepath, array_name="az", verbose=False
    )

    # There is currently a bug in MAYA4 downloader utils that skips the last row and col of chunks for some products. I will solve that later, the quick fix is to clamp.
    # Determine effectively available extent (intersection of RCMC and AZ)
    eff_az_stop = min(available_RCMC_az_slice.stop, available_AZ_az_slice.stop)
    eff_rg_stop = min(available_RCMC_rg_slice.stop, available_AZ_rg_slice.stop)

    if eff_az_stop < full_az_size or eff_rg_stop < full_rg_size:
        log_info(
            f"WARNING: Dataset is incomplete. Processing restricted to available extent: {eff_az_stop}x{eff_rg_stop} (Full: {full_az_size}x{full_rg_size})"
        )
        full_az_size = eff_az_stop
        full_rg_size = eff_rg_stop
    else:
        log_info(f"Dataset is complete. Full size: {full_az_size}x{full_rg_size}")

    # ---------------------------------------------------------
    log_step("Defining and loading Patch and Region of Interest (ROI)")

    # I have manually selected interesting patches for each downloaded file
    patch_coordinates_per_file = {
        "05d521": (3000, 12000),  # Weird image, default: (3000, 12000)
        "05e631": (7000, 5000),  # TODO (7000, 5000)
        "0685cc": (7000, 5000),  # Image contains a simple island (7000, 5000)
    }
    PATCH_SIZE = 2000
    patch_az_start = patch_coordinates_per_file[filename_shorthand][0]
    patch_az_end = patch_az_start + PATCH_SIZE
    patch_rg_start = patch_coordinates_per_file[filename_shorthand][1]
    patch_rg_end = patch_rg_start + PATCH_SIZE

    patch_az_slice = slice(patch_az_start, patch_az_end)
    patch_rg_slice = slice(patch_rg_start, patch_rg_end)
    patch_coordinates_str = f"{patch_az_start}:{patch_az_end}_{patch_rg_start}:{patch_rg_end}"
    patch_rcmc = p.get_array("rcmc")[patch_az_slice, patch_rg_slice]
    patch_az = p.get_array("az")[patch_az_slice, patch_rg_slice]
    log_info(
        f"Loaded Patch for file {filename_shorthand} at azimuth [{patch_az_start}:{patch_az_end}], range [{patch_rg_start}:{patch_rg_end}]."
    )

    # However, for the compression we need the entire azimuth stripes, so the ROI contains the patch in range but full azimuth
    roi_az_start = 0
    roi_az_end = full_az_size
    roi_rg_start = patch_rg_start
    roi_rg_end = patch_rg_end

    # Override to full image in range if requested
    if args.full_image:
        log_info(f"Processing Full Image {args.full_image=}. Overriding ROI to full range.")
        roi_rg_start = 0
        roi_rg_end = full_rg_size

    roi_az_slice = slice(roi_az_start, roi_az_end)
    roi_rg_slice = slice(roi_rg_start, roi_rg_end)
    roi_rcmc = p.get_array("rcmc")[roi_az_slice, roi_rg_slice]  # Loaded in Time-Domain
    log_info(
        f"Loaded ROI for processing at azimuth [{roi_az_start}:{roi_az_end}], range [{roi_rg_start}:{roi_rg_end}]."
    )

    # ---------------------------------------------------------
    log_step("Initial Visualization")

    utils.save_image_subsampled(
        VIS_DIR / f"whole_rcmc.{PLOT_FORMAT}",
        np.abs(full_rcmc),
        title="Complete RCMC",
        target_size=500,
        patch_box=(patch_az_slice, patch_rg_slice),
    )
    full_az = p.get_array("az")
    utils.save_image_subsampled(
        VIS_DIR / f"whole_az.{PLOT_FORMAT}",
        np.abs(full_az),
        title="Complete AZ",
        target_size=500,
        patch_box=(patch_az_slice, patch_rg_slice),
    )
    utils.save_image_subsampled(
        VIS_DIR / f"patch_rcmc_{patch_coordinates_str}.{PLOT_FORMAT}",
        np.abs(patch_rcmc),
        title="Patch RCMC",
        target_size=max(patch_rcmc.shape),
    )
    utils.save_image_subsampled(
        VIS_DIR / f"patch_az_{patch_coordinates_str}.{PLOT_FORMAT}",
        np.abs(patch_az),
        title="Patch AZ",
        target_size=max(patch_az.shape),
    )
    del full_az, patch_az, patch_rcmc

    # ---------------------------------------------------------
    log_step("Adapting Metadata for Slice")

    # We load full tables, but we need to pass appropriate metadata to CoarseRDA
    # CoarseRDA expects metadata to match the data dimensions implicitly via params like SWST
    metadata_full = utils.load_partitioned_dataframe(p, "metadata")
    ephemeris_full = utils.load_partitioned_dataframe(p, "ephemeris")

    # When we slice the RCMC data in range (data[:, start:end]), the new 0-th column
    # corresponds to column 'start' in the original data.
    # The Processor calculates the Slant Range (Distance) for each pixel based on
    # Sampling Window Start Time (SWST), which is the time of the very first sample.
    # If we don't adjust SWST, the processor will think our sliced data starts at the
    # original Near Range, leading to incorrect Range Reference Function generation and RCMC.
    # Therefore, we shift SWST by the time duration of the samples we skipped.

    metadata_slice = metadata_full.copy()

    # Slice Metadata in Azimuth to match the available/selected RCMC data
    # (Essential if we are processing a partial image or strict ROI)
    if roi_az_slice.start > 0 or roi_az_slice.stop < full_az_size:
        log_info(f"Slicing metadata to match ROI Azimuth: {roi_az_slice}")
        metadata_slice = metadata_slice.iloc[roi_az_slice].reset_index(drop=True)

    if roi_rg_start > 0:  # ignore if full image or starting at 0
        log_info(f"Applying Range Offset for sliced data starting at sample {roi_rg_start}")
        # Get Range Sampling Frequency
        meta_row = metadata_slice.iloc[0]
        rgdec = meta_row.get("Range Decimation", meta_row.get("range_decimation"))
        if rgdec is None:
            raise ValueError(
                "Could not find 'Range Decimation' in metadata to compute range frequency."
            )
        range_freq = float(RANGE_DECIMATION_MAP[int(rgdec)])  # Hz
        print(f"  -> Range Sampling Frequency: {range_freq} Hz (Decimation: {rgdec})")

        # Calculate time offset for the start of our range slice
        time_offset = roi_rg_start / range_freq
        log_info(f"  -> Applying Range Offset: {roi_rg_start} samples -> {time_offset:.9f} s")

        # Update SWST in metadata
        swst_col = "swst"
        old_swst = float(metadata_slice.iloc[0][swst_col])
        metadata_slice[swst_col] = metadata_slice[swst_col] + time_offset
        new_swst = float(metadata_slice.iloc[0][swst_col])
        log_info(f"Updated {swst_col}: {old_swst} -> {new_swst}")
    else:
        log_info("No Range Offset applied (Processing from sample 0 or Full Image).")

    # ---------------------------------------------------------
    log_step("Initializing CoarseRDA")

    # We construct a mock 'raw_data' dictionary to initialize the processor, that stores it as self.radar_data.
    mock_raw_data = {"echo": roi_rcmc, "metadata": metadata_slice, "ephemeris": ephemeris_full}

    # Initialize Processor, verbose=True will print steps from sarpyx
    processor = CoarseRDA(mock_raw_data, verbose=True, memory_efficient=False)

    # OPTIMIZATION: Free memory now that processor took ownership (or copied)
    del roi_rcmc
    del mock_raw_data
    gc.collect()

    # ---------------------------------------------------------
    log_step("Processing: Azimuth Compression")

    # The CoarseRDA pipeline usually expects the data to be in the Range-Doppler domain
    # (Range Time, Azimuth Frequency) before Azimuth Compression step.
    # Our loaded RCMC data is in Time-Time domain, so we must perform Azimuth FFT first.

    log_info("1. Transforming RCMC to Doppler Domain (Azimuth FFT)...")
    # We use fft without fftshift to recover the original centered spectrum (since ifft was done without shift)
    processor.radar_data = np.fft.fft(processor.radar_data, axis=0)

    log_info("2. Computing Effective Velocities...")
    # This populates internal parameters needed for the D term calculation
    processor._compute_effective_velocities()

    # 2b. Manually Compute D parameter (normally done in get_rcmc)
    log_info("2b. Manually Computing D Parameter...")
    processor.wavelength = TX_WAVELENGTH_M

    # Generate centered azimuth frequency values
    # Exact code from get_rcmc() in sarpyx
    processor.az_freq_vals = np.arange(
        -processor.az_sample_freq / 2,
        processor.az_sample_freq / 2,
        processor.az_sample_freq / processor.len_az_line,
    )
    # Ensure we have exactly the right number of frequency values
    if len(processor.az_freq_vals) != processor.len_az_line:
        processor.az_freq_vals = np.linspace(
            -processor.az_sample_freq / 2,
            processor.az_sample_freq / 2,
            processor.len_az_line,
            endpoint=False,
        )
    log_info(f"Azimuth frequency values shape: {processor.az_freq_vals.shape}")
    log_info(f"Effective velocities shape: {processor.effective_velocities.shape}")

    # D calculation
    # Reduce effective velocity to 1D (per azimuth line) as in sarpyx
    mean_effective_velocities = np.mean(processor.effective_velocities, axis=1)
    log_info(f"Mean effective velocities shape: {mean_effective_velocities.shape}")

    # D = sqrt( 1 - (lambda*f)^2 / (4*V^2) )
    processor.D = np.sqrt(
        1
        - (processor.wavelength**2 * processor.az_freq_vals**2)
        / (4 * mean_effective_velocities**2)
    )
    log_info(f"D (cosine squint angle) shape: {processor.D.shape}")

    log_info("3. Performing Azimuth Compression (Filter * IFFT)...")
    # This function:
    #   - Generates/Gets Azimuth Filter (using velocities and slant range)
    #   - Multiplies radar_data * filter
    #   - Performs IFFT on Azimuth axis
    processor._perform_azimuth_compression_efficient()

    # Get Result, already in time domain
    roi_az_recon = processor.radar_data

    # Calculate Local Slices for the Patch (relative to the ROI processed)
    local_patch_az_slice = slice(patch_az_start - roi_az_start, patch_az_end - roi_az_start)
    local_patch_rg_slice = slice(patch_rg_start - roi_rg_start, patch_rg_end - roi_rg_start)

    patch_az_recon = np.abs(roi_az_recon[local_patch_az_slice, local_patch_rg_slice])

    # ---------------------------------------------------------
    log_step("Validation & Saving")

    # Save full compressed ROI
    utils.save_image_subsampled(
        VIS_DIR
        / f"ROI_compressed_sarpyx_{roi_az_start}:{roi_az_end}_{roi_rg_start}:{roi_rg_end}.{PLOT_FORMAT}",
        np.abs(roi_az_recon),
        title="ROI azimuth compressed with Sarpyx",
        target_size=500,
    )
    # Save patch from the compressed data for closer inspection
    utils.save_image_subsampled(
        VIS_DIR / f"patch_compressed_sarpyx_{patch_coordinates_str}.{PLOT_FORMAT}",
        patch_az_recon,
        title=f"Patch azimuth compressed with Sarpyx [{patch_az_start}:{patch_az_end}, {patch_rg_start}:{patch_rg_end}]",
        target_size=max(patch_az_end - patch_az_start, patch_rg_end - patch_rg_start),
    )

    # ---------------------------------------------------------
    log_step("Correlation Analysis")

    # Cropping MARGIN_AZ from top and bottom to avoid edge effects in correlation
    MARGIN_AZ = 1000
    if roi_az_recon.shape[0] < (2 * MARGIN_AZ):
        raise ValueError("Compressed data too small for margin cropping.")
    log_info(f"Cropping {MARGIN_AZ} pixels margin for correlation.")
    corr_az_slice = slice(MARGIN_AZ, -MARGIN_AZ)

    corr_az_recon = roi_az_recon[corr_az_slice, :]

    # Load Ground Truth Slice - MEMORY OPTIMIZED
    # Instead of loading the full ROI and then slicing it, we calculate the exact slice of the origin array we need.
    # roi_az_slice covers roi_az_start to roi_az_end.
    # We want (roi_az_start + MARGIN) to (roi_az_end - MARGIN)

    log_info("Loading azimuth slice ('Ground Truth') for correlation...")

    # Calculate global start/stop for the correlation area
    corr_global_slice = slice(roi_az_slice.start + MARGIN_AZ, roi_az_slice.stop - MARGIN_AZ)
    corr_az_gt = p.get_array("az")[corr_global_slice, roi_rg_slice]

    corr_c = utils.compute_correlation(corr_az_recon, corr_az_gt)
    log_info(f"Sarpyx correlation (complex): {corr_c:.4f}")
    corr_m = utils.compute_mag_correlation(corr_az_recon, corr_az_gt)
    log_info(f"Sarpyx correlation (magnitude): {corr_m:.4f}")

    del corr_az_recon, corr_az_gt
    gc.collect()

    # --- COMPARISON PLOT ---
    log_info("Generating comparison plot...")

    # Reload original patches (magnitude)
    patch_rcmc = np.abs(p.get_array("rcmc")[patch_az_slice, patch_rg_slice])
    patch_az = np.abs(p.get_array("az")[patch_az_slice, patch_rg_slice])

    plot_path = VIS_DIR / f"sarpyx_az_compression_comparison.{PLOT_FORMAT}"

    utils.save_comparison_plot(
        path=plot_path,
        images=[patch_rcmc, patch_az, patch_az_recon],
        titles=["Input RCMC (Mag)", "Ground Truth AZ (Mag)", "Sarpyx Output (Mag)"],
        main_title=f"Sarpyx Azimuth Compression Comparison (full-resolution, area: {patch_az_start}:{patch_az_end}, {patch_rg_start}:{patch_rg_end})",
    )

    t_end = time.time()
    log_info(f"Total Time: {t_end - t_start_total:.2f} s")


if __name__ == "__main__":
    main()
