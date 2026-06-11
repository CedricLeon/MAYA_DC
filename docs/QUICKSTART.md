# MAYA_DC — Quick Start Guide

**All commands must be run inside the `MAYA_DC` conda environment.**

---

## ⚡ TL;DR

```bash
conda activate MAYA_DC

# Start training
python src/train.py experiment=rcmc_compress_baseline

# Quick smoke-test (2 batches, no GPU needed)
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2
```

---

## 📋 Prerequisites

### 1 · Environment

Package versions live in `pyproject.toml`. `environment.yaml` pins the Python
interpreter and delegates all pip installs to `pyproject.toml` in one shot.
Run from the **repo root**:

> **Dependencies install automatically** — no manual cloning of `maya4`/`sarpyx`.
> `maya4` and `compressai` install from git (fixes not yet on PyPI; see
> `pyproject.toml`), `sarpyx` from PyPI. A C/C++ compiler is needed for `compressai`.

```bash
conda env create -f environment.yaml
conda activate MAYA_DC
python -c "import torch, lightning, hydra, compressai, maya4, sarpyx; print('All OK')"
```

To update an existing env after `pyproject.toml` or `environment.yaml` changes:

```bash
conda env update -f environment.yaml --prune
```

> **Without conda:** `pip install -e ".[dev]"` from the repo root installs
> everything (pulling `maya4`/`compressai` from git, `sarpyx` from PyPI) into
> whatever Python environment is currently active.

### 2 · Data

MAYA4 products live in the **public HuggingFace bucket `ESA-philab/Maya4`** (maya4's
default). Two ways to feed the model:

- **Directory-based** (the baseline experiment): point `data.train_dir/val_dir/test_dir`
  at local folders of zarr products.
- **Bucket streaming** (filter-based datamodule): `data=maya4 data.online=true` streams
  chunks on demand from the bucket. Verified working:

```bash
python src/train.py experiment=rcmc_compress_baseline data=maya4 data.online=true \
  '~data.train_dir' '~data.val_dir' '~data.test_dir'
```

> **Filter caveat:** the bucket currently holds `vv` / 2025 S1–S2 products, while the
> default `configs/data/maya4.yaml` filters select `hh,hv` (the paper's selection) — so
> widen `data.polarizations` / `data.years` / `data.stripmap_modes` to match what the
> bucket actually contains, otherwise the filters match zero products.

---

## 🎯 Training

### Baseline experiment

```bash
python src/train.py experiment=rcmc_compress_baseline
```

Key overrides:

| Override | Example | Effect |
|----------|---------|--------|
| `data.online` | `data.online=true` | Stream missing zarr chunks from HuggingFace |
| `data.max_products_train` | `data.max_products_train=5` | Fewer products (faster epoch) |
| `data.samples_per_prod` | `data.samples_per_prod=20` | Patches sampled per product (0 = all) |
| `data.batch_size` | `data.batch_size=2` | Reduce if GPU OOM |
| `model.criterion.lmbda` | `model.criterion.lmbda=10` | Rate–distortion λ (paper sweep `1…1000`, baseline `1000`). Lower λ → higher compression |
| `trainer.max_epochs` | `trainer.max_epochs=50` | Training duration |
| `logger=tensorboard` | | Enable TensorBoard logging |

### Debugging

```bash
# 2 batches only (fastest sanity check)
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2

# Hydra fast-dev-run preset
python src/train.py debug=fdr

# Overfit on a single batch
python src/train.py debug=overfit
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## ⚙️ Key Config Parameters

### `configs/data/maya4.yaml` (and experiment overrides)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | `[512, 512]` | Core patch size (azimuth, range) |
| `azimuth_buffer` | `512` | Buffer lines added on each side for azimuth focusing context |
| `samples_per_prod` | `0` | Patches sampled per product per epoch (0 = all) |
| `online` | `false` | Stream from the HF bucket `ESA-philab/Maya4` |
| `max_products_train` | `50` | Max training products |
| `max_products_val` | `5` | Max validation products |
| `batch_size` | `4` | Batch size |

> **Divisibility constraint:** `patch_size[0] + 2 × azimuth_buffer` must be
> divisible by 16.  The datamodule raises an `AssertionError` on startup if
> this is violated.  Valid combinations: `patch_az=512, buf=128 → 768`,
> `patch_az=2048, buf=512 → 3072`.

### `configs/data/maya4_dir.yaml` (directory-based splits)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_dir` | `null` | Directory with training zarr products (recursive scan) |
| `val_dir` | `null` | Directory with validation zarr products |
| `test_dir` | `null` | Directory with test zarr products |
| `max_products_train/val/test` | `-1` | Cap on products per split after ephemeris check (`-1` = all) |
| `samples_per_prod` | `0` | Patches per product per epoch (`0` = all) |

Usage:

```bash
python src/train.py experiment=rcmc_compress_baseline data=maya4_dir \
  data.train_dir=data/splits/train data.val_dir=data/splits/val \
  data.patch_size=[512,512] data.azimuth_buffer=512
```

### `configs/model/rcmc_compress.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nb_input_channels` | `2` | Real + imaginary SAR channels |
| `nb_channels_main` | `128` | Feature channels N in the encoder/decoder |
| `nb_channels_latent` | `null` | Latent dim M (null → 2 N) |
| `activation` | `"gdn"` | Non-linearity (`"gdn"` or `"relu"`) |
| `criterion.lmbda` | `1000` (baseline) | Rate–distortion λ; override via `model.criterion.lmbda` (paper sweep `1,3,5,7,8,10,15,20,35,50,100,1000`) |
| `delta_kde` | `1.0` | Weight for KDE distribution term |
| `delta_coherence` | `1.0` | Weight for phase coherence term |
| `azimuth_buffer` | `512` | Must match `data.azimuth_buffer` |
| `training_mode` | `"slc"` | `"slc"`: train in SLC domain via azimuth compression; `"rcmc"`: train in RCMC domain, no azimuth compression (F8) |

---

## 🏗 Project Layout

```
src/
  train.py / eval.py              ← Hydra entry points
  data/maya4_datamodule.py        ← filter-based datamodule (parts/years; HF bucket)
  data/maya4_dir_datamodule.py    ← directory-based datamodule (baseline uses this)
  models/rcmc_compress_module.py  ← training LightningModule (RCMCDCmodule)
  models/components/scale_hyperprior.py, losses.py
  utils/sarpyx_azimuth_compression.py  ← differentiable az compression (numpy H + torch.fft)

scripts/
  validate_azimuth_pipeline.py    ← numerical check: custom FFT vs CoarseRDA (F10)
  evaluate_classical_codec_rd.py  ← JPEG/JPEG2000/WebP RD baselines
  visualize_data.py               ← data-pipeline diagnostic
  experiment_block_processing.py  ← azimuth buffer-size study
  estimate_model_ops.py           ← model MACs / FLOPs / params

notebooks/
  RD-curve_plots.ipynb            ← paper RD curves (reads paper_results/)
  plot_combined_rd_curves.py, make_overview_figure.py
  paper_results/                  ← tracked plot-input CSVs

configs/
  experiment/rcmc_compress_baseline.yaml · model/rcmc_compress.yaml
  data/{maya4,maya4_dir}.yaml · training_mode/{slc,slc_mse,rcmc}.yaml
  fixed_patches/                  ← patch spec for the opt-in MonitorFixedPatch callback
```

---

## 📋 See Also

- [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) — bug-fix/feature history (F1–F15), *historical*
- [../CONVERGENCE_ANALYSIS.md](../CONVERGENCE_ANALYSIS.md) — convergence diagnosis, *historical*
- [cluster.md](cluster.md) — how the paper runs were launched (ESA SpaceHPC / PBS)
- [raw-data.md](raw-data.md) — off-git paper-data layout
- [../progress_tracking.md](../progress_tracking.md) — research TODO list, *historical*
