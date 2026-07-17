# MAYA_DC: Early Compression of SAR Data Pre-Focusing

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://huggingface.co/buckets/ESA-philab/Maya4"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ESA--philab%2FMaya4-ffcc00.svg"></a>
<!-- TODO: add the paper/DOI badge once the EUSAR26 proceedings are published, e.g.:
[![Paper](http://img.shields.io/badge/DOI-xxxxx-B31B1B.svg)](https://doi.org/xxxxx) -->

</div>

## Description

This project uses Learned Image Compression (LIC) networks to compress Range Cell Migration Corrected (RCMC) SAR data.
The results were submitted and presented at [EUSAR26](https://www.eusar.de/en).

### Motivation

This project explores the possibility to compress Synthetic Aperture Radar (SAR) data during the focusing pipeline.
SAR data consists of complex-valued radar echoes.
When acquired this RAW data is equivalent to processing Level 0.
Through various signal processing steps, these radar echoes are assembled to construct an image of the observed scene called Single-Look Complex (SLC), this image is equivalent to Level 1.
Efficiently compressing SAR SLC data is a complex but achievable task.
However, it requires to construct (focus) the SLC onboard the data collection platform which is typically a resource-constrained environment (SmallSats or UAVs).
Ideally the compression of the data would be done on the RAW data (as performed nowadays by conventional codecs such as BAQ, or FDBAQ), but it is an extremely complex task given the size and nature of these arrays of echoes.

In an effort to push the learned-compression of SAR data as early as possible in the processing pipeline, we perform compression of RCMC data.
The focusing of SAR data, i.e., the transformation from L0 to L1, can be summarized in 3 steps:

1. Range focusing: `raw` (L0) to `rc`
2. Range Correction: `rc` to `rcmc`
3. Azimuth Compresson: `rcmc` to `az` (L1)

In this project, we compressed data at the `rcmc` stage: a LIC model encodes/decodes the RCMC data, then a differentiable azimuth-compression step focuses the reconstruction into an SLC that is compared against the ground-truth SLC.
The detailed problem formulation and the azimuth-filter
derivation are in the EUSAR26 paper (see [Citation](#citation)).

### Dataset & Method

The data is the public [`ESA-philab/Maya4`](https://huggingface.co/buckets/ESA-philab/Maya4) HuggingFace bucket.
It consists of four representations (from `raw` to `az`) of Sentinel-1 products and can be streamed directly at train time
(no manual download; see [Quick start](#quick-start)).

As for the method, we used a simple hyperprior autoencoder based on [Variational image compression with a scale hyperprior](https://openreview.net/forum?id=rkcQFMZRb) (Ballé et al., 2018).

### Results

...

## Usage

### Installation

All Python dependencies are declared in `pyproject.toml` and install automatically — including the SAR packages `maya4` and `sarpyx` (**no manual cloning needed**).
`environment.yaml` pins the Python interpreter (3.12) and runs a single `pip install -e ".[dev]"`.

> **Note:** `maya4` and `compressai` currently install from git (the fixes we need aren't on PyPI yet — see the comments in `pyproject.toml`); `sarpyx` installs from PyPI. A C/C++ compiler must be available to build `compressai`.

```bash
# 1. Clone the repo
git clone https://github.com/CedricLeon/MAYA_DC
cd MAYA_DC

# 2. Create the environment (run from repo root)
conda env create -f environment.yaml
conda activate MAYA_DC
# Use `pip install -e ".[dev]"` to do the same without conda

# 3. Verify
python -c "import torch, lightning, hydra, compressai, maya4, sarpyx; print('All OK')"
```

> **CI note:** the automated **test** and **code-coverage** workflows are temporarily disabled. Code-quality (pre-commit) checks still run. See `.github/workflows/test.yml` to re-enable.

### Quick start

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for the full operator reference including training commands, key config parameters, and the project layout.

```bash
# Smoke-test (CPU, 2 batches)
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2

# Full training run (set data.{train,val,test}_dir to local zarr dirs)
python src/train.py experiment=rcmc_compress_baseline

# …or stream the dataset from the public HF bucket (ESA-philab/Maya4)
python src/train.py experiment=rcmc_compress_baseline data=maya4 data.online=true
```

The entry points are `src/train.py` (training) and `src/eval.py` (evaluation).
All parameters are managed by [Hydra](https://hydra.cc/); so you can override anything from the CLI:

```bash
python src/train.py experiment=rcmc_compress_baseline \
  model.criterion.lmbda=10 \
  data.batch_size=2 \
  trainer.max_epochs=100
```

For debugging:

```bash
python src/train.py debug=fdr      # fast_dev_run, no logging
python src/train.py debug=overfit  # overfit on 1 batch
```

### Reproducing the paper results

The paper compares the learned codec (square vs non-square kernels) against classical codecs (JPEG / JPEG2000 / WebP) across a rate–distortion sweep of the $\lambda$ knob.

```bash
# 1. Train the sweep — λ is the rate–distortion trade-off (paper sweep):
for L in 1 3 5 7 8 10 15 20 35 50 100 1000; do
  python src/train.py experiment=rcmc_compress_baseline model.criterion.lmbda=$L
done
# Note: You should also train for several seeds, for example we trained for 3 seeds in the paper.

# 2. Classical-codec rate–distortion baselines. One CSV per codec (codec ∈ {jpeg, jpeg2000, webp}):
python scripts/evaluate_classical_codec_rd.py <products_dir> jpeg2000 baselines_jpeg2000.csv

# 3. Rate–distortion curves (reads the tracked CSVs under notebooks/paper_results/):
python notebooks/plot_combined_rd_curves.py

# 4. Qualitative reconstruction overview (needs the raw .npy reconstructions under data/from_cluster/, see docs/raw-data.md):
python notebooks/make_overview_figure.py
```

The pre-computed inputs used for the paper figures are tracked under [`notebooks/paper_results/`](notebooks/paper_results); the canonical interactive analysis is [`notebooks/RD-curve_plots.ipynb`](notebooks/RD-curve_plots.ipynb).

## Citation

<!-- TODO: add a CITATION.cff (and BibTeX below) once the EUSAR26 proceedings are published. -->

If you use this code or the associated results, please cite the EUSAR26 paper (reference to be added once the proceedings are published).
