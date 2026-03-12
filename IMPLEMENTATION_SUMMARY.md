# MAYA_DC — Implementation Summary

**Date:** March 2026
**Status:** ✅ Pipeline implemented and gradient-verified (10-epoch run, no null/zero grads). Loss convergence is the active open problem.

---

## 🎯 Project Goal

Compress Sentinel-1 **Range Cell Migration Corrected (RCMC)** SAR data using a
Neural Image Compression (NIC) model, then reconstruct a **Single-Look Complex
(SLC)** image by azimuth compression of the decompressed RCMC, and compare it
against the ground-truth SLC.

```
RCMC (B,2,Az+2·buf,Rg)
         │
    ┌────▼────────────────┐
    │  ScaleHyperprior    │  Neural compression/decompression
    │  (NIC encoder/dec.) │  + entropy coding (likelihoods)
    └────┬────────────────┘
         │  RCMC_hat (B,2,Az+2·buf,Rg)
    ┌────▼────────────────┐
    │  Azimuth Compress   │  CoarseRDA matched filter (sarpyx)
    │  (strip buffer)     │  or identity FFT/IFFT (fallback)
    └────┬────────────────┘
         │  SLC_hat (B,2,Az,Rg)
         │
    Compare with GT SLC ──► Loss: Rate + λ·Distortion
```

---

## 📦 Code Map

| File | Role |
|------|------|
| `src/data/maya4_datamodule.py` | `LightningDataModule`; loads MAYA4 zarr patches with metadata |
| `src/models/rcmc_compress_module.py` | `LightningModule`; training/validation loop |
| `src/models/components/scale_hyperprior.py` | NIC model (CompressAI ScaleHyperprior, 2-channel SAR input) |
| `src/models/components/losses.py` | `SimpleMSELoss`, `CompoundCompressionLoss`, `CompoundSARLoss`; helpers: `estimate_likelihoods_bpp`, `kde_histogram_loss`, `complex_coherence_loss` |
| `src/utils/azimuth_compression.py` | Identity azimuth compression (FFT→IFFT, preserves phase) |
| `src/utils/sarpyx_azimuth_compression.py` | Standalone differentiable azimuth compression; filter H computed in numpy (mirrors CoarseRDA math), applied via `torch.fft` so gradients flow back through the decoder |
| `scripts/sarpyx_azimuth_compression.py` | Stand-alone sarpyx validation script |
| `configs/data/maya4.yaml` | Hydra data config |
| `configs/model/rcmc_compress.yaml` | Hydra model config |
| `configs/experiment/rcmc_compress_baseline.yaml` | Baseline experiment config |

---

## 🔧 Bug Fixes Applied

### BUG 1 — `simple_azimuth_compress` called `.abs()` (phase destroyed)
**File:** `src/utils/azimuth_compression.py`
**Fix:** Replaced with `azimuth_compress_with_metadata`, which converts real/imag
channels to a complex tensor, applies FFT→IFFT in the complex domain, strips the
azimuth buffer, then returns real/imag channels **without calling `.abs()`**.
`simple_azimuth_compress` is kept with a `DeprecationWarning`.

### BUG 2 — Metadata never plumbed through the data pipeline
**File:** `src/data/maya4_datamodule.py`
**Fix:** Added `MetadataAwareSARZarrDataset` (subclasses `SARZarrDataset`) which
returns `(patch_from, patch_to, zfile, y, x)` per item.  A custom collate fn
assembles these into 5-tuples.  `_build_metadata_for_patch` loads, azimuth-slices,
and applies the SWST range offset from `RANGE_DECIMATION_MAP`.  Metadata is cached
in `self._metadata_cache` to avoid redundant zarr attr reads.

### BUG 3 — `online` hardcoded `True` in data loader
**Files:** `src/data/maya4_datamodule.py`, `configs/data/maya4.yaml`
**Fix:** Added `online: bool = False` parameter; passed through to `SARZarrDataset`.
Set `online: false` in configs to avoid unintentional HuggingFace downloads.

### BUG 4 — No `samples_per_prod` parameter (confused with `max_products`)
**Files:** `src/data/maya4_datamodule.py`, `configs/data/maya4.yaml`,
`configs/experiment/rcmc_compress_baseline.yaml`
**Fix:** Added `samples_per_prod: int = 0` (0 = all patches).  Passed to both
`SARZarrDataset` and `KPatchSampler`.

### BUG 5 — `from_state_dict` reconstructed wrong channel counts
**File:** `src/models/components/scale_hyperprior.py`
**Fix:**
- `nb_input_channels = state_dict["g_a.0.weight"].size(1)` (was `.size(0)` → gave N, not C_in)
- `nb_channels_main  = state_dict["g_a.0.weight"].size(0)`
- `nb_channels_latent = state_dict["g_a.6.weight"].size(0)`
- GDN detection changed from `"g_a.1.weight"` (doesn't exist for GDN) to
  presence of `"g_a.1._beta"` (a GDN-specific parameter).

### BUG 6 — Hydra `rcmc_compress.yaml` self-referenced itself
**File:** `configs/model/rcmc_compress.yaml`
**Fix:** Removed the circular `defaults: [- override /model: rcmc_compress.yaml]`
block that caused Hydra to resolve the config infinitely.

---

## 🛠 Functional Fixes

### FIX 7 — No assertion that `(patch_az + 2·buffer) % 16 == 0`
**File:** `src/data/maya4_datamodule.py`
**Fix:** Added assert in `MAYA4DataModule.__init__`.  The model has 4 stride-2
conv layers → 16× spatial downsampling.  Patches whose height is not a multiple
of 16 cause shape mismatches in the entropy bottleneck.

### FIX 8 — `kde_histogram_loss` OOM on large patches
**File:** `src/models/components/losses.py`
**Fix:** Replaced the Gaussian KDE (which built an `(N × num_bins)` tensor —
OOM at `N = 4M` pixels) with `torch.histc` — O(N) memory cost.

### FIX 9 — Global coherence loss was meaningless (LLN effect)
**File:** `src/models/components/losses.py`
**Fix:** Replaced global complex correlation with a **windowed coherence** using
`F.avg_pool2d` over `window_size=7` spatial windows.  Returns `1 - mean(γ_local)`.

### FIX 10 — Training step method named `some_function` (placeholder)
**File:** `src/models/rcmc_compress_module.py`
**Fix:** Renamed to `_extract_inputs_targets`; handles both legacy 2-tuple and
new 5-tuple batch format.

### FIX 11 — LR scheduler `step()` called twice per validation epoch
**File:** `src/models/rcmc_compress_module.py`
**Fix:** Removed `sch.step()` from `training_step`; kept only in
`on_validation_epoch_end` with `ReduceLROnPlateau.step(val_loss)`.

### FIX 14 — bpp normalised by SLC pixels instead of RCMC pixels
**File:** `src/models/components/losses.py` (all four loss classes)
**Fix:** Replaced `N,C,H,W = x_hat.shape` (SLC after buffer trim) with
`B,_,Hy,Wy = likelihoods["y"].shape; num_pixels = B*(Hy*16)*(Wy*16)` to
recover the RCMC input pixel count by undoing the 16× encoder downsampling.

### BUG 15 — Zero gradient to `g_s` decoder (CoarseRDA numpy break)
**File:** `src/utils/sarpyx_azimuth_compression.py`
**Fix:** Replaced the CoarseRDA-based numpy loop with a standalone implementation.
The azimuth filter H is computed in numpy/scipy (same maths as CoarseRDA, but
no full processor instantiation), converted to a constant torch tensor, and
applied via `torch.fft` / `torch.fft.ifft`. The data path (`FFT(x̂) × H → IFFT`)
stays entirely in PyTorch so `d(SLC)/d(x̂) = IFFT(H)` is propagated correctly.
`torch.view_as_complex` / `torch.view_as_real` are used instead of
`torch.complex()` / `.real`/`.imag` for reliable autograd across non-contiguous
channel slices.

### BUG 16 — `[grad-check]` false alarm on `entropy_bottleneck.quantiles`
**File:** `src/models/rcmc_compress_module.py`
**Fix:** Params ending in `.quantiles` are intentionally skipped in the grad-check
loop — they belong to `aux_optimizer` and only receive gradients from
`manual_backward(aux_loss)`, never from `criterion["loss"]`.

### BUG 17 — Spurious gradient through `dx` in `kde_histogram_loss`
**File:** `src/models/components/losses.py`
**Fix:** `dx = (...).detach()` — the bin-width constant should not contribute
to gradients; without detach it dragged `ps.max()`/`ps.min()` into the graph.

---

## ✨ Improvements

### IMPROVE 12 — Expose `nb_channels_latent` as independent parameter
**File:** `src/models/components/scale_hyperprior.py`
**Change:** Added `nb_channels_latent: Optional[int] = None`; defaults to
`2 * nb_channels_main` (the original behaviour).  Allows independent control
of the hyper-prior latent space dimension.

### IMPROVE 13 — Identity model validation script *(planned, not yet created)*
A `scripts/validate_azimuth_pipeline.py` is referenced in QUICKSTART but not yet
implemented.  It would load a zarr patch, run identity (FFT/IFFT) and the
standalone sarpyx filter, and compare both against the ground-truth SLC.

---

## 🏗 Data Flow Details

### Batch format
Each dataloader batch is a 5-tuple:

```python
(
    rcmc,          # Tensor (B, 2, Az+2·buf, Rg) float32  — real/imag channels
    slc,           # Tensor (B, 2, Az+2·buf, Rg) float32  — real/imag channels
    metadata_list, # List[pd.DataFrame | None]  — azimuth-sliced, SWST-corrected
    ephemeris_list,# List[pd.DataFrame | None]
    coords_list,   # List[{"zfile": str, "y": int, "x": int}]
)
```

### Buffer convention
- MAYA4 patches are loaded with `patch_size = (patch_az + 2·buffer, patch_rg)`.
- After azimuth compression the buffer rows are stripped → output is `(B,2,patch_az,Rg)`.
- **Constraint:** `patch_az + 2·buffer` must be divisible by 16 (enforced by assertion).

### Azimuth compression in training
`rcmc_compress_module._azimuth_compress`:
1. Tries `full_azimuth_compress_batch` (sarpyx) if metadata is available.
2. Falls back to `azimuth_compress_with_metadata` (identity FFT/IFFT) otherwise.

---

## ⚙️ Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.patch_size` | `[2048, 2048]` | Core patch size (azimuth, range) |
| `data.azimuth_buffer` | `512` | Azimuth buffer lines on each side |
| `data.samples_per_prod` | `0` | Patches per product per epoch (0 = all) |
| `data.online` | `false` | Stream data from HuggingFace |
| `data.max_products_train` | `10` | Max training products |
| `model.net.nb_channels_main` | `128` | Encoder feature channels (N) |
| `model.net.nb_channels_latent` | `null` | Hyper-prior latent dim (null = 2N) |
| `model.lmbda` | `0.01` | Rate-distortion trade-off λ |
