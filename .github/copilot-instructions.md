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
      - Source code in `src/`; standalone scripts in `scripts/` (e.g., `validate_azimuth_pipeline.py`, which replicates the SAR focusing pipeline (RCMC $\to$ SLC) via `sarpyx` to validate compression quality); analysis in `notebooks/`;
      - All configs are in `configs/` and tests in `tests/`;
      - Data resides in `data/` (gitignored); runs/results under `logs/`;
      - `maya4` (dataset) and `sarpyx` (SAR processing) install automatically from `pyproject.toml` (maya4 + compressai from git, sarpyx from PyPI) — no local clones.

## Critical Workflows & Commands
*Every command that require packages should be run inside the `MAYA_DC` conda environment.*
- **Training**:
    ```bash
    python src/train.py experiment=rcmc_compress_baseline
    ```
- **Azimuth Compression (Verification)**:
    - Use `scripts/validate_azimuth_pipeline.py` to check the differentiable focusing vs sarpyx CoarseRDA.
    ```bash
    python scripts/validate_azimuth_pipeline.py --input_file <product>.zarr
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

### 5. SAR Data Visualization Conventions (Critical)
- **Display**: Always use **log-intensity**: `logI = ln(re² + im² + ε)` from `phys_to_logI_torch`.
  Clip for contrast: `mean ± clip_factor·std` (default `clip_factor=3.0`).
  Always use the **`viridis`** colormap. Never display raw real/imag channels or linear amplitude directly.
- **Orientation**: **(0, 0) is at the top-left corner**. Azimuth is the **vertical axis, top → bottom** (Az=0 at the top edge). Range is the **horizontal axis, left → right** (Rg=0 at the left edge). Implementation: data shape is `(Az, Rg)`; display directly without transposing — matplotlib rows = azimuth (vertical), cols = range (horizontal). Use `origin="upper"` and `ax.set_xlabel("Range →")` / `ax.set_ylabel("Azimuth ↓")`. Use `ax.set_facecolor("black")` so non-downloaded chunks appear black.
- **Metrics**: Always compute on **linear amplitude**: `|·| = sqrt(re² + im²)` from `phys_to_linA_torch`,
  or on the normalised `[0, 1]` domain.
  **Never compute PSNR, SSIM, coherence, or KDE on log-scale data.**
- **Normalised domain `[0, 1]`**: RCMC channels normalized with `RC_MIN=-3000, RC_MAX=3000`;
  SLC channels with `GT_MIN=-12000, GT_MAX=12000`.
  After the BUG 27 fix, both re/im channels should sit in `[0, 1]` after the dataloader.

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

## `# type: ignore` Policy

Use `# type: ignore[<code>]` **only** when the type error is genuinely
unfixable with the current stubs:

| Pattern | Comment tag | Reason |
| :--- | :--- | :--- |
| `pl_module.hparams.some_attr` | `# type: ignore[attr-defined]` | Lightning hparam stubs expose `hparams` as `dict \| Namespace`; attribute access is always flagged. |
| `pl_module.logger.experiment.log(...)` | `# type: ignore[union-attr]` | Lightning's `logger` type is a union; the concrete WandB logger is only resolved at runtime. |

**Never** use `# type: ignore` to paper over real type mismatches.
Prefer narrowing types explicitly with `isinstance` checks or explicit casts.
Avoid bare `# type: ignore` without an error code.

## Special characters
- Do not use the special character `–` (U+2013), instead use the traditional - (U+002D) for dashes in text.
- Similarly, do not use `×` (U+00D7) but `x` (U+0078).
