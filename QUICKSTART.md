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

```bash
conda env create -f environment.yaml
conda activate MAYA_DC
python -c "import torch, lightning, hydra, maya4, sarpyx; print('All OK')"
```

### 2 · Data

MAYA4 zarr products must already be downloaded locally (set `online: false` in
configs).  To trigger a one-off download of missing chunks set `data.online=true`
on the command line:

```bash
python src/train.py experiment=rcmc_compress_baseline data.online=true
```

After the first run the data is cached; switch back to `online: false`.

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
| `model.lmbda` | `model.lmbda=0.001` | Lower λ → higher compression ratio |
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
| `patch_size` | `[2048, 2048]` | Core patch size (azimuth, range) |
| `azimuth_buffer` | `512` | Buffer lines added on each side for azimuth focusing context |
| `samples_per_prod` | `0` | Patches sampled per product per epoch (0 = all) |
| `online` | `false` | Stream data from HuggingFace |
| `max_products_train` | `500` | Max training products |
| `max_products_val` | `50` | Max validation products |
| `batch_size` | `4` | Batch size |

> **Divisibility constraint:** `patch_size[0] + 2 × azimuth_buffer` must be
> divisible by 16.  The datamodule raises an `AssertionError` on startup if
> this is violated.  Valid combinations: `patch_az=512, buf=128 → 768`,
> `patch_az=2048, buf=512 → 3072`.

### `configs/model/rcmc_compress.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nb_input_channels` | `2` | Real + imaginary SAR channels |
| `nb_channels_main` | `128` | Feature channels N in the encoder/decoder |
| `nb_channels_latent` | `null` | Latent dim M (null → 2 N) |
| `activation` | `"gdn"` | Non-linearity (`"gdn"` or `"relu"`) |
| `lmbda` | `0.01` | Rate-distortion trade-off λ |
| `loss_type` | `"mse"` | Loss class (`"mse"`, `"compound"`) |

---

## 🏗 Project Layout

```
src/
  train.py                        ← training entry point
  eval.py                         ← evaluation entry point
  data/
    maya4_datamodule.py           ← MAYA4 LightningDataModule
  models/
    rcmc_compress_module.py       ← training LightningModule
    components/
      scale_hyperprior.py         ← NIC model
      losses.py                   ← loss functions
  utils/
    azimuth_compression.py        ← identity azimuth compression (FFT→IFFT)
    sarpyx_azimuth_compression.py ← differentiable azimuth compression;
                                     H computed in numpy (CoarseRDA maths),
                                     applied via torch.fft — gradients flow
                                     back through the decoder

scripts/
  sarpyx_azimuth_compression.py   ← stand-alone sarpyx compression demo

configs/
  data/maya4.yaml
  model/rcmc_compress.yaml
  experiment/rcmc_compress_baseline.yaml
```
