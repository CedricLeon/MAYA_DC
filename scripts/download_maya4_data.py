import functools
import os
import sys
from pathlib import Path

import maya4
import numpy as np
from maya4 import (
    GT_MAX,
    GT_MIN,
    RC_MAX,
    RC_MIN,
    SampleFilter,
    SARTransform,
    get_sar_dataloader,
    minmax_normalize,
)

# Print version
print(f"Maya4 version: {maya4.__version__}")

# ============================================================================
# 📁 STEP 1: DATA DIRECTORY SETUP
# ============================================================================
# Define where SAR products are stored locally (or will be downloaded to)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_complete_download"
print(f"Storing/Downloading data to: {DATA_DIR}")

# makedir if doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# 🎛️ STEP 2: SAMPLE FILTERING (What data to load?)
# ============================================================================
# SampleFilter allows you to select specific SAR products based on metadata

filters = SampleFilter(
    # years: List of acquisition years to include (e.g., [2023, 2024])
    # Only products from these years will be loaded
    years=[2023],
    # months: List of acquisition months to include
    # We filter for June to target the specific product '20230616'
    months=[6],
    # polarizations: Radar polarization modes to include
    # Options: "hh", "hv", "vh", "vv" (H=horizontal, V=vertical)
    # "hh" = horizontal transmit, horizontal receive (co-pol)
    # "hv" = horizontal transmit, vertical receive (cross-pol)
    polarizations=["hh"],
    # stripmap_modes: Sentinel-1 stripmap beam modes (1-6)
    # Different modes have different swath widths and resolutions
    # Modes 1-3: narrow swaths, higher resolution
    # Modes 4-6: wider swaths, lower resolution
    stripmap_modes=[1, 2, 3],
    # parts: Geographic regions/partitions in the dataset
    # Maya4 organizes data into parts (PT1, PT2, etc.) by location
    # Leave empty or None to include all parts
    parts=["PT4"],
)

# ============================================================================
# 🔄 STEP 3: TRANSFORMATION PIPELINE (How to normalize data?)
# ============================================================================
transforms = SARTransform(
    # transform_raw: Applied to Level 0 (raw) data
    # Raw data typically has different statistics than compressed data
    transform_raw=functools.partial(
        minmax_normalize,
        array_min=RC_MIN,  # Minimum value for normalization
        array_max=RC_MAX,  # Maximum value for normalization
    ),
    # transform_rc: Applied to Range Compressed data
    # RC is the first compression step (range direction only)
    transform_rc=functools.partial(minmax_normalize, array_min=RC_MIN, array_max=RC_MAX),
    # transform_rcmc: Applied to Range Cell Migration Corrected data
    # RCMC corrects for range cell migration due to platform motion
    transform_rcmc=functools.partial(minmax_normalize, array_min=RC_MIN, array_max=RC_MAX),
    # transform_az: Applied to Azimuth focused (final) data
    # This is the ground truth - fully focused SAR image
    transform_az=functools.partial(
        minmax_normalize, array_min=GT_MIN, array_max=GT_MAX  # Different min/max for focused data
    ),
)

# ============================================================================
# 🎯 STEP 4: CREATE THE DATALOADER (Main configuration)
# ============================================================================
print("=" * 80)
print("🔧 Creating Maya4 SAR Dataloader with comprehensive configuration...")
print("=" * 80 + "\n")

loader = get_sar_dataloader(
    data_dir=str(DATA_DIR),
    online=True,  # Enable downloading
    # Levels RCMC to AZ
    level_from="rcmc",
    level_to="az",
    # Download settings modified to avoid OOM
    return_whole_image=False,  # CHANGED: Avoid loading full image to memory
    samples_per_prod=0,  # CHANGED: Iterate over all patches
    # Patch configuration optimized for download
    patch_mode="rectangular",
    patch_size=(2048, 2048),  # Reasonable patch size
    stride=(2048, 2048),  # Non-overlapping strides to cover everything once
    patch_order="chunk",  # CHANGED: Use chunk order for optimized download
    max_products=5,  # Get a few products to ensure we find target
    # Ensure Zarr v2
    zarr_version=2,  # Argument we added to get_sar_dataloader
    use_balanced_sampling=False,  # Disable balanced sampling to avoid API rate limits
    # Transforms & Filters
    transform=transforms,
    filters=filters,
    # Batch settings
    batch_size=1,
    num_workers=0,
    verbose=True,
)

# Iterate to trigger download
print("✓ Dataloader configured")
print("Iterating through loader to trigger downloads...")

count = 0
for i, (x_batch, y_batch) in enumerate(loader):
    count += 1
    if i % 10 == 0:
        print(f"Processed patch {i}: shape {x_batch.shape}")

print(f"Download complete! Processed {count} patches.")
