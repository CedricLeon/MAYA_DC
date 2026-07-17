# CLAUDE.md — MAYA_DC working notes

> Orientation for agents/maintainer. If something here disagrees with the code,
> trust the code and fix this file.

## Commit Workflow

Before committing Python changes, run `black .` (and the other pre-commit hooks) first, then
stage the reformatted files — otherwise the formatter rewrites them and blocks the first commit.

## Editing Files

Read a file immediately before editing it so exact-match strings are current; for large or
repetitive blocks, read the surrounding context first to avoid match failures.

## What this is

Neural Image Compression (NIC) of **Range Cell Migration Corrected (RCMC)** Sentinel-1 SAR
data — compressing *before* full focusing. The decoded RCMC is azimuth-compressed
(differentiably) into an SLC and compared to the ground-truth SLC. Submitted to **EUSAR26**.
Stack: PyTorch Lightning + Hydra (from `lightning-hydra-template`) + CompressAI.

Pipeline (`src/models/rcmc_compress_module.py`):
```
RCMC (B,2,Az+2·buf,Rg) → ScaleHyperprior (compress+reconstruct) → x̂
    → full_azimuth_compress_batch (torch FFT, differentiable) → SLC_recon → loss(SLC_recon, SLC_target)
```
Two `training_mode`s: `slc` (default, full pipeline) and `rcmc` (F8 — compare x̂ directly to
RCMC, no az compression; for products lacking ephemeris or debugging).

## Code map

- `src/train.py`, `src/eval.py` — Hydra entry points.
- `src/models/rcmc_compress_module.py` — `RCMCDCmodule` (manual optimization: net + aux
  optimizer for the entropy bottleneck; logs per-module grad norms + RD scatter to WandB).
- `src/models/components/scale_hyperprior.py` — the NIC net. Supports **non-square kernels**
  (scale factor hardcoded to 3 — open TODO to derive from patch_size).
- `src/models/components/losses.py` — compound loss + metrics (`complex_correlation_metric`,
  `psnr_amplitude`, `ssim_amplitude`, `phase_preservation_metric`).
- `src/data/maya4_datamodule.py` — filter-based (parts/years) datamodule over the MAYA4 HF
  interface (streams from the bucket when `online=true`).
- `src/data/maya4_dir_datamodule.py` — **directory-based** datamodule (the baseline uses
  this); scans dirs of zarr products. Yields `(rcmc, slc, metadata, ephemeris, coords)`.
- `src/utils/sarpyx_azimuth_compression.py` — differentiable azimuth compression: filter `H`
  built in numpy (CoarseRDA maths), applied via `torch.fft`. Module-level `_FILTER_CACHE`.
- `src/callbacks/` — `monitor_val_reconstruction.py` (on by default) and
  `monitor_fixed_patch.py` (**opt-in** — needs a local product; see `configs/fixed_patches/`).
- Analysis lives in `notebooks/`: `RD-curve_plots.ipynb` (canonical), `plot_combined_rd_curves.py`,
  `make_overview_figure.py`, with tracked plot-input CSVs in `notebooks/paper_results/`.
- Standalone scripts in `scripts/`: `validate_azimuth_pipeline.py` (F10 numerical check),
  `visualize_data.py`, `evaluate_classical_codec_rd.py` (JPEG/JPEG2000/WebP baselines),
  `experiment_block_processing.py`, `estimate_model_ops.py`.

Config: Hydra in `configs/`. Loss/mode via the `training_mode` group (`slc` | `slc_mse` | `rcmc`).
Main experiment: `experiment=rcmc_compress_baseline`. **λ** is the rate–distortion knob, set via
`model.criterion.lmbda` (paper sweep `1,3,5,7,8,10,15,20,35,50,100,1000`; baseline default `1000`).

## Dependencies (de-vendored 2026-06)

Installable from a fresh clone — no manual cloning. In `pyproject.toml`:
- `sarpyx` — PyPI (`>=0.1.10`).
- `maya4` — **git `pypi` branch** (PyPI 0.1.2 is unusable: zarr-v2 `zarr.hierarchy` + no HF-bucket
  support). `TODO(maya4-pypi)` to switch once upstream publishes.
- `compressai` — **git `main`** (PyPI 1.2.8 caps `numpy<2`). `TODO(compressai-pypi)` likewise.
- `huggingface-hub>=1.5` (bucket API). `numpy>=2`. Building `compressai` from git needs a
  C/C++ compiler. Detailed investigation: `docs/dependency-bump-notes.md`.

## data/ (gitignored: `/data/`)

- The canonical dataset is the **public HF bucket `ESA-philab/Maya4`** (maya4's default);
  online streaming is verified working.
- `data/from_cluster/` (off-git) — paper artifacts: `results_extracted/` (raw `.npy`
  reconstructions) + `cache3_extracted/` (codec-eval outputs). See `docs/raw-data.md`.
  Paper experiment = `sq` vs `nsq` kernels x the λ sweep x seeds, vs JPEG/JPEG2000/WebP.

## Commands

```bash
conda activate MAYA_DC                                                  # env: environment.yaml → pyproject.toml
python src/train.py experiment=rcmc_compress_baseline                   # full run (set data.*_dir to local zarr dirs)
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2   # smoke test
python src/train.py experiment=rcmc_compress_baseline data=maya4 data.online=true  # stream from the HF bucket
python src/train.py debug=fdr            # fast-dev-run preset
```
Constraint: `patch_size[0] + 2*azimuth_buffer` must be divisible by 16 (datamodule asserts).

## Cluster context

Paper runs used **ESA SpaceHPC** with **PBS/OpenPBS** (`qsub`/`qstat`). The PBS launchers were
removed; their commands are preserved in **`docs/cluster.md`**.
