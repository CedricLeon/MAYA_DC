import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------
def log_step(name):
    """Logs a step name with surrounding dashes for clarity."""
    print(f"\n{'-'*5} {name} {'-'*5}")


def log_info(msg):
    """Logs an informational message with indentation."""
    print(f"    {msg}")


# ---------------------------------------------------------
# Metadata/Ephemeris Fix Helper
# ---------------------------------------------------------
def load_partitioned_dataframe(handler, attr_name):
    """Manually reconstruct a DataFrame from Zarr attributes."""
    # Access private method to get store
    store = handler._load_store()
    if attr_name not in store.attrs:
        raise KeyError(f"No '{attr_name}' key found in Zarr attributes.")

    raw_data = store.attrs[attr_name]

    if not isinstance(raw_data, dict):
        return pd.DataFrame(raw_data)

    partitions = []
    for key, records in raw_data.items():
        if key == "data":
            idx = 0
        elif key.startswith("data_slice_"):
            try:
                idx = int(key.split("_slice_")[-1])
            except ValueError:
                continue
        else:
            continue
        partitions.append((idx, records))

    partitions.sort(key=lambda x: x[0])
    all_records = [record for idx, records in partitions for record in records]

    if not all_records:
        raise ValueError(f"No records found in {attr_name} partitions.")
    try:
        return pd.DataFrame(all_records)
    except Exception as e:
        raise ValueError(f"Error creating DataFrame from records for {attr_name}: {e}") from e


def scan_available_data_extent(
    filepath: Path, array_name: str = "rcmc", verbose: bool = False
) -> tuple[slice, slice]:
    """Scans the Zarr directory to find the extent of the available chunks and returns a tuple of
    slices (az_slice, range_slice) covering the valid area."""
    root = filepath / array_name

    zarr_version = None
    if (root / ".zarray").exists():
        zarr_version = "v2"
    elif (root / "zarr.json").exists():
        zarr_version = "v3"
    else:  # zarr_version is None
        raise ValueError(f"Could not determine Zarr version for array '{array_name}' at {root}.")

    if verbose:
        log_info(f"Scanning directory: {root}, detected Zarr version: {zarr_version}.")

    if zarr_version == "v2":
        with open(root / ".zarray") as f:
            meta = json.load(f)
            chunk_rows = meta["chunks"][0]
            chunk_cols = meta["chunks"][1]
            full_shape = meta["shape"]
    else:  # "v3"
        with open(root / "zarr.json") as f:
            meta = json.load(f)
            chunk_rows = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
            chunk_cols = meta["chunk_grid"]["configuration"]["chunk_shape"][1]
            full_shape = meta["shape"]

    if verbose:
        log_info(
            f"Found full array shape: {full_shape}, with chunk size: {chunk_rows}x{chunk_cols}."
        )

    # Calculate expected chunks with ceiling division (equivalent math.ceil())
    expected_row_chunks = -(full_shape[0] // -chunk_rows)
    expected_col_chunks = -(full_shape[1] // -chunk_cols)
    if verbose:
        log_info(
            f"Expected chunks layout: {expected_row_chunks} rows x {expected_col_chunks} cols = {expected_row_chunks * expected_col_chunks} chunks total."
        )

    if zarr_version == "v2":
        # Look for files like "0.0", "1.2", etc.
        chunk_files = [f for f in root.iterdir() if f.is_file() and not f.name.startswith(".")]
        if verbose:
            log_info(
                f"  -> Found {len(chunk_files)} chunk files, first: {chunk_files[0].name} and last: {chunk_files[-1].name}."
            )

        found_chunks = []
        for f in chunk_files:
            try:
                parts = f.name.split(".")
                # Zarr V2 Default separator is '.' : "row.col"
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    row = int(parts[0])
                    col = int(parts[1])
                    found_chunks.append((row, col))
            except ValueError:
                continue
    else:  # "v3"
        c_path = root / "c"
        if not c_path.exists():
            raise ValueError(
                f"Expected 'c' directory for Zarr v3 at {c_path}, but it does not exist."
            )

        # Look for row directories (integers)
        # Structure assumed: root/c/ROW/COL
        row_dirs = [p for p in c_path.iterdir() if p.is_dir() and p.name.isdigit()]
        found_chunks = []

        for r_dir in row_dirs:
            r_idx = int(r_dir.name)
            # Look for col files/dirs (integers)
            col_files = [
                p for p in r_dir.iterdir() if (p.is_file() or p.is_dir()) and p.name.isdigit()
            ]
            for c_file in col_files:
                c_idx = int(c_file.name)
                found_chunks.append((r_idx, c_idx))

        if not found_chunks:
            raise ValueError(f"No valid chunks found in Zarr v3 structure at {c_path}.")

    # Remove duplicates from found_chunks (otherwise continuity check will fail, as we find as many '0' row chunks as there are columns)
    found_chunks.sort()
    all_row_chunks = list({row for row, _ in found_chunks})
    all_col_chunks = list({col for _, col in found_chunks})

    # Check for missing chunks against expectation
    if verbose:
        missing_chunks = []
        for r in range(expected_row_chunks):
            for c in range(expected_col_chunks):
                if (r, c) not in found_chunks:
                    missing_chunks.append((r, c))

        if missing_chunks:
            log_info(f"WARNING: Missing chunks: {missing_chunks}")
        else:
            log_info("All expected chunks are valid/present.")

    # Verify that all the chunks are continuous in rows and columns
    def is_continuous(my_list: list[int]) -> bool:
        return all(a + 1 == b for a, b in zip(my_list, my_list[1:]))

    if not is_continuous(all_row_chunks) or not is_continuous(all_col_chunks):
        log_info(f"Chunks are not continuous. Found chunks: {found_chunks}")
        raise ValueError(
            f"Chunks are not continuous. Row chunks: {all_row_chunks}. Column chunks: {all_col_chunks}."
        )
    if verbose:
        log_info("All chunks are continuous in both rows and columns.")

    # 3. Calculate range (Azimuth)
    first_row_chunk = found_chunks[0][0]
    last_row_chunk = found_chunks[-1][0] + 1
    first_col_chunk = found_chunks[0][1]
    last_col_chunk = found_chunks[-1][1] + 1
    if verbose:
        log_info(
            f"Found chunk range - Rows: {first_row_chunk} to {last_row_chunk}, Cols: {first_col_chunk} to {last_col_chunk}."
        )

    # Calculate coverage in pixels
    az_start = first_row_chunk * chunk_rows
    az_stop = last_row_chunk * chunk_rows
    rg_start = first_col_chunk * chunk_cols
    rg_stop = last_col_chunk * chunk_cols

    # CLAMP to actual image shape because valid chunks at the edge might extend beyond the image shape (padding)
    az_stop = min(az_stop, full_shape[0])
    rg_stop = min(rg_stop, full_shape[1])

    az_slice = slice(az_start, az_stop)
    rg_slice = slice(rg_start, rg_stop)

    if verbose:
        log_info(f"  -> Azimuth Slice: {az_slice}, Range Slice: {rg_slice}")

    return az_slice, rg_slice


# ---------------------------------------------------------
# Mathematical Utilities
# ---------------------------------------------------------


def compute_correlation(a, b):
    """Compute the normalized complex correlation."""
    return np.abs(np.sum(a * np.conj(b))) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_mag_correlation(a, b):
    """Compute the Pearson correlation of the magnitudes."""
    return np.corrcoef(np.abs(a).flatten(), np.abs(b).flatten())[0, 1]
