# MAYA_DC: Early Compression of SAR Data Pre-Focusing

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This project uses Neural Image Compression (NIC) networks to compress Range Cell Migration Corrected (RCMC) SAR data.
The results were submitted and presented at [EUSAR26](https://www.eusar.de/en).

### Motivation

This project explores the possibility to compress Synthetic Aperture Radar (SAR) data during the focusing pipeline.
SAR data consists of complex-valued radar echoes when acquired, this RAW dat is equivalent to processing Level 0 (L0).
Through various signal processing steps, these radar echoes are assembled to construct an image of the observed seen called Single-Look Complex (SLC), this image is equivalent to Level 1 (L1).
Efficiently compressing SAR SLC data is a complex but achievable task. However, it requires to construct the SLC onboard the data collection platform which is a resource-constrained environment (SmallSats or UAVs), ideally the compression of the data would be done on the RAW data which is a extremely complex task given the size and nature of these arrays of echoes. Nowadays, this task is still performed by conventional codecs such as BAQ, or FDBAQ.

In an effort to push the learned-compression of SAR data as early as possible in the processing pipeline, we perform compression of RCMC data.
The focusing of SAR data, i.e., the transformation from L0 to L1, can be summarized in 3 steps:

1. Range focusing: `raw` (L0) to `rc`
2. Range Correction: `rc` to `rcmc`
3. Azimuth Compresson: `rcmc` to `az` (L1)

## Problem formulation

*Extracted from my Obsidian vault.*

### Manual Azimuth Compression

We have the original (or reconstructed) *rcmc* image and want to compute the *az* image.
To do so, we need to work in Fourier domain and compute the *azimuth filter*. Following the notations of Rich-Hall in his notebook:

$$\text{Azimuth filter} = exp\biggl\{4i\pi\frac{R_{0}D(f_{\eta}, V_{r})}{\lambda}\biggl\}$$
Where:

- $R_0$ is the *slant range of closest approach*: The straight-line distance from the satellite track to each range bin on the ground. It is constant for all azimuth lines and has the shape $(N_{rg},)$, a 1D vector where $N_{rg}$ is the number of range samples.
- $D$ is the migration factor, or the cosinus of the instantaneous squint angle, as it varies with both azimuth and range it has the same shape as our radar data. It is calculated with:
  - $f_{\eta}$, the frequency axis after the FFT
  - $V_{r}$ is the effective spacecraft velocity

In the code Rich-Hall compute the migration factor per chunk for an easier memory management (see the `yield` keyword in [[Python]] to transform a function in iterator).

**Problem** (see question 1 of [[Onboarding Phi-Lab meeting Roberto - 2026-01-08]]): the chunks are only in azimuth directions. They contain the complete range arrays ... Does it work if we only have part of it?

- If **yes** then we don't even need the chunking, the data should be small enough to fit entirely in memory and we can perform azimuth compression in one step.

Because we cannot process the complete image $A$ we manipulate it as *patches*, i.e., contiguous subsets of $A$ called *submatrices* or $A_S$. The question is how do we calculate the equivalent subset of $C$ that we call $C_S$.

## Installation

All Python dependencies are declared in `pyproject.toml` and install
automatically — including the SAR packages `maya4` and `sarpyx` (**no manual
cloning needed**). `environment.yaml` pins the Python interpreter (3.12) and runs
a single `pip install -e ".[dev]"`.

> **Note:** `maya4` and `compressai` currently install from git (the fixes we
> need aren't on PyPI yet — see the comments in `pyproject.toml`); `sarpyx`
> installs from PyPI. A C/C++ compiler must be available to build `compressai`.

```bash
# 1. Clone the repo
git clone <repo_url>
cd MAYA_DC

# 2. Create the environment (run from repo root)
conda env create -f environment.yaml
conda activate MAYA_DC

# 3. Verify
python -c "import torch, lightning, hydra, compressai, maya4, sarpyx; print('All OK')"
```

> **Without conda:** `pip install -e ".[dev]"` from the repo root works too — it
> installs everything, pulling `maya4`/`compressai` from git and `sarpyx` from PyPI.

To update after pulling new changes:

```bash
conda env update -f environment.yaml --prune
```

### Optional: pre-commit hooks

```bash
pre-commit install
```

> **CI note:** the automated **test** and **code-coverage** workflows are
> temporarily disabled. Code-quality (pre-commit) checks still run. See
> `.github/workflows/test.yml` to re-enable.

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for the full operator reference including
training commands, key config parameters, and the project layout.

```bash
# Smoke-test (CPU, 2 batches)
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2

# Full training run (set data.{train,val,test}_dir to local zarr dirs)
python src/train.py experiment=rcmc_compress_baseline

# …or stream the dataset from the public HF bucket (ESA-philab/Maya4)
python src/train.py experiment=rcmc_compress_baseline data=maya4 data.online=true
```

## Usage

The entry points are `src/train.py` (training) and `src/eval.py` (evaluation).
All parameters are managed by [Hydra](https://hydra.cc/); override anything from
the CLI:

```bash
python src/train.py experiment=rcmc_compress_baseline \
  model.criterion.lmbda=10 \
  data.batch_size=2 \
  trainer.max_epochs=100
```

For debugging:

```bash
python src/train.py debug=fdr   # fast_dev_run, no logging
python src/train.py debug=overfit  # overfit on 1 batch
```
