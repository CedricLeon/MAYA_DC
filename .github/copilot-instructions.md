# MAYA_DC: AI Coding Assistant Instructions

## Agent instructions
- Before answering, wrap your step-by-step reasoning inside <thinking> tags.
- Never use jargon. Never write sentences above 20 words. Never assume technical knowledge.

## Project Context
**MAYA_DC** (Early Compression of SAR Data Pre-Focusing) implements Neural Image Compression (NIC) on Range Cell Migration Corrected (RCMC) SAR data.
The codebase is a hybrid of deep learning (PyTorch Lightning + Hydra) and signal processing (SAR focusing/reconstruction).

## "Big Picture" Architecture
- **Framework**: Built on `lightning-hydra-template`.
    - **Entry Points**: `src/train.py` (Training), `src/eval.py` (Evaluation).
    - **Configuration**: Hydra-based in `configs/`. *Do not hardcode parameters*; use YAML configs or command-line overrides.
    - **Modules**:
        - `src/models/`: LightningModules (Neural Networks).
        - `src/data/`: LightningDataModules (Data Loaders).
- **Data Flow**:
    - **Input**: RCMC SAR data in **Zarr** format (Complex `complex128` IQ data).
    - **Pipeline**:
      - Compress RCMC data using NIC models create $\hat{RCMC}$, compare input RCMC to $\hat{RCMC}$ and also perform Azimuth Compression on $\hat{RCMC}$ to generate $\hat{SLC}$ and compare SLC and $\hat{SLC}$.
    - **Project Structure**:
      - Source code in `src/` and standalone scripts in `scripts/` (e.g., `sarpyx_azimuth_compression.py` that replicate the SAR focusing pipeline (RCMC $\to$ SLC) via `sarpyx` to validate compression quality.);
      - All configs are in `configs/` and tests should be written in `tests/`;
      - Data resides in `data/`; while runs and results are stored in `logs/`;
      - Local packages used for the projects are stored in `Maya4/` (MAYA4) and `srp/` (sarpyx).

## Critical Workflows & Commands
*Every command that require packages should be run inside the `MAYA_DC` conda environment.*
- **Training**:
    ```bash
    python src/train.py experiment=example
    ```
- **Azimuth Compression (Verification)**:
    - Use `scripts/sarpyx_azimuth_compression.py` for to experiment with the azimuth focusing pipeline.
    ```bash
    python scripts/sarpyx_azimuth_compression.py --input_file data/PT4/sample.zarr
    ```
- **Debugging**:
    - Use `debug=fdr` (or other) in Hydra commands.
    ```bash
    python src/train.py debug=fdr
    ```

## Project-Specific patterns & Conventions

### 1. Memory Management (Critical)
SAR images are massive (e.g., 25k x 25k pixels, Complex128).
- **Never load full arrays**: Do not use `[:]` on Zarr arrays, instead use lazy loading with `slice()` objects:.
```python
# BAD
full_data = zarr_array[:]
# GOOD
roi_slice = (slice(0, 1000), slice(0, 1000))
patch = zarr_array[roi_slice]
```
- **Garbage Collection**: Explicitly `del large_array` and run `gc.collect()` in tight loops (especially in `scripts/`).
- **Zarr chunks**: Zarr arrays are chunked and it is possible that not all chunks have been downloaded. Always check for continuity and if necessary clamp the extent to the image shape.
```python
count_row = min(last_row_chunk * chunk_rows, full_rows)
```

### 4. Path Handling
- Use `rootutils` to ensure paths are relative to the project root, regardless of where the script is run.
    ```python
    import rootutils
    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    ```

## External Dependencies
- **Hydra**: Config management.
- **PyTorch Lightning**: Training loop abstraction.
- **Sarpyx**: Internal/Library for SAR metadata and sensor models.
- **Maya4**: Internal/Library used for loadeing the MAYA4 datasets.

## Working with Documentation Files

The project uses three living markdown files that must be kept up to date:

### [`IMPLEMENTATION_SUMMARY.md`](../IMPLEMENTATION_SUMMARY.md)
The technical memory of the project. Update it whenever:
- A bug is fixed: add a `BUG N` or `FIX N` entry in the existing style.
- A new feature is implemented: move it from the Future Features table to a new `IMPROVE N` entry.
- A design decision is made: add a note to the **Design Notes** section.
- A new planned feature is identified: add a row to the **Future Features** table (F1–FN).

### [`QUICKSTART.md`](../QUICKSTART.md)
The operator's reference. Update it whenever:
- A command changes or a new entry point is added.
- A config parameter is added, renamed, or its default changes.
- A key file is added or removed from the project layout.

### [`progress_tracking.md`](../progress_tracking.md)
**Maintained by the user.** Do not rewrite sections or restructure this file.
You may only:
- Check off a `[ ]` item to `[x]` when the user confirms a task is done.
- Add a new `- [ ]` item if the user explicitly asks for it.

### General rule
After any non-trivial code change, update the relevant section of
`IMPLEMENTATION_SUMMARY.md` in the same response. Do not defer documentation to
a separate step.
