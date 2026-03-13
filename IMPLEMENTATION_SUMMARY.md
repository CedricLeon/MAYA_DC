# MAYA_DC — Implementation Summary

**Date:** March 2026
**Status:** ✅ Pipeline implemented and gradient-verified (10-epoch run, no null/zero grads). Loss convergence is the active open problem.

---

## 🎯 Project Goal

Compress Sentinel-1 **Range Cell Migration Corrected (RCMC)** SAR data using a
Neural Image Compression (NIC) model, then reconstruct a **Single-Look Complex
(SLC)** image by azimuth compression of the decompressed RCMC, and compare it
against the ground-truth SLC.

```txt
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
| :--- | :--- |
| `src/data/maya4_datamodule.py` | `LightningDataModule`; loads MAYA4 zarr patches with metadata |
| `src/models/rcmc_compress_module.py` | `LightningModule`; training/validation loop |
| `src/models/components/scale_hyperprior.py` | NIC model (CompressAI ScaleHyperprior, 2-channel SAR input) |
| `src/models/components/losses.py` | `SimpleMSELoss`, `CompoundCompressionLoss`, `CompoundSARLoss`; helpers: `estimate_likelihoods_bpp`, `kde_histogram_loss`, `complex_coherence_loss` |
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

### BUG 25 — `online=False` crashes on metadata-only Zarr stores

**File:** `Maya4/maya4/dataloader.py`
**Root cause:** In *online* mode `_initialize_stores` skips store opening entirely
(lazy access on demand); incomplete files never trigger an error.  In *offline*
mode it calls `_append_file_to_stores` for every file, which calls `open_archive`,
which raises `RuntimeError` when a Zarr group is empty (metadata downloaded but
no data chunks).  The `except` block re-raised that error instead of skipping.
**Fix:** Changed the `except Exception` clause (for both `zarr` and `dask`
backends) to print a `[WARN]` message and drop the file from `self._files`
instead of re-raising.  Training proceeds with the complete files that are
actually available locally.  The warning output (`[WARN] Skipping '…': could
not open store offline (metadata-only download?). …`) is always visible (not
gated on `verbose`) so users can identify which products need re-downloading.

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

### IMPROVE 13 — Azimuth pipeline validation script (F10)

**File:** `scripts/validate_azimuth_pipeline.py`
**Change:** Created a standalone comparison script with no network involved.
Loads the same RCMC patch and compares three results side-by-side:

- **GT SLC** — the `az` product from the MAYA4 zarr (ground truth).
- **CoarseRDA** — the full sarpyx `CoarseRDA` processor (reference algorithm).
- **Custom FFT** — `full_azimuth_compress_batch` from
  `src/utils/sarpyx_azimuth_compression.py` (no network, same code path as training).

Metrics reported for each pair: complex correlation, magnitude correlation, PSNR [dB],
SSIM.  Outputs two PNG plots: magnitude comparison and residual error maps.

**Validation results** (`patch_az=3000, patch_rg=12000, patch_size=1024`):

| Metric | CoarseRDA vs GT | Custom FFT (buf=512) vs GT | Custom FFT (buf=1024) vs GT |
| :--- | ---: | ---: | ---: |
| Complex corr | 1.0000 | 0.9761 | 0.9998 |
| Magnitude corr | 1.0000 | 0.9574 | 0.9996 |
| PSNR [dB] | 73.01 | 36.55 | 56.55 |
| SSIM | 1.0000 | 0.9594 | 0.9995 |

**Key finding:** `buffer=512` is insufficient (~97% correlation); `buffer=1024` gives
>99.9% correlation and >56 dB PSNR.  The `azimuth_buffer` config default should be
`1024` (or at minimum `512` is acceptable if memory is tight, but reconstruction
quality is noticeably lower).

### IMPROVE 18 — Quality metrics at validation: complex correlation, PSNR, SSIM (F2, F3)

**Files:** `src/models/components/losses.py`, `src/models/rcmc_compress_module.py`
**Change:**

- Added three standalone metric functions to `losses.py`:
  - `complex_correlation_metric(pred, target)` — returns `(mean, std)` of
    per-image global coherence $|\langle\hat{s}, s^*\rangle| / (\|\hat{s}\|\cdot\|s\|)$.
  - `psnr_amplitude(pred, target)` — PSNR [dB] on amplitude images with
    per-image `data_range = max(|target|)`.
  - `ssim_amplitude(pred, target)` — SSIM on amplitude images via
    `torchmetrics.functional.image.structural_similarity_index_measure`.
- `forward_with_az_compression` now returns a 3-tuple
  `(loss_dict, slc_recon, slc_target)` instead of just `loss_dict`.
  Training step discards the tensors; validation and test steps use them.
- `validation_step` and `test_step` log four additional scalars per epoch:
  `valid/complex_corr_mean`, `valid/complex_corr_std`, `valid/psnr_amp`, `valid/ssim_amp`
  (and the matching `test/` variants).

### IMPROVE 19 — Per-module gradient norm logging (F5)

**File:** `src/models/rcmc_compress_module.py`
**Change:** After `manual_backward(criterion["loss"])` and before `clip_gradients`,
`training_step` now iterates over `g_a`, `g_s`, `h_a`, `h_s` and logs
`train/grad_norm_<module>` (total L2 norm of all parameter gradients in that
module) every step.  Logged pre-clip so the raw signal is preserved.

### IMPROVE 20 — Colorbars + physical SLC value range in `MonitorValReconstruction` (F1)

**File:** `src/callbacks/monitor_val_reconstruction.py`
**Change:**

- Each subplot row now uses a **shared `vmin`/`vmax`** across all columns (pre-computed
  before `imshow`), so all patches in a row are on the same intensity scale.
- A **per-row colorbar** is added to the right of each row via
  `fig.colorbar(im_ref, ax=axes[row_idx, :], shrink=0.7, pad=0.02)`, giving a
  precise logI value scale for each row type.
- **Physical min/max** of `|slc_recon|` and `|slc_target|` (before log transform) are
  computed and embedded in the figure `suptitle` (`SLC recon |·| ∈ [min, max]`) and
  logged as `val_batch/slc_recon_phys_max` and `val_batch/slc_target_phys_max` to
  WandB, enabling automatic detection of scale explosions across training.

### IMPROVE 21 — Rate–Distortion scatter plot in WandB (F4)

**File:** `src/models/rcmc_compress_module.py`
**Change:** Added `self._rd_table` (lazy `wandb.Table`) in `__init__`.  Each
`on_validation_epoch_end` appends a `(epoch, rate_bpp, distortion)` row and
logs `valid/rd_scatter` via `wandb.plot.scatter`.  The table grows over time
so the full training trajectory is visible in a single WandB panel.

### IMPROVE 22 — `x_hat` histogram to WandB (F6)

**File:** `src/callbacks/monitor_val_reconstruction.py`
**Change:** Added `val_batch/x_hat_histogram: wandb.Histogram(output.x_hat.ravel())`
to the existing WandB log call at the end of each monitored validation batch.
Displays the full value distribution of the decoder output, making collapse
(all zeros) or saturation (values at `±1`) immediately visible.

### IMPROVE 23 — RCMC-domain training mode (F8)

**Files:** `src/models/rcmc_compress_module.py`, `configs/model/rcmc_compress.yaml`
**Change:**

- Added `forward_no_az_compression` method to `RCMCCompressModule`.  Trims the
  azimuth buffer from `x_hat` and compares it directly to `rcmc_target`
  (normalised RCMC core), skipping `full_azimuth_compress_batch` entirely.
- Added `_forward_step` dispatch method: routes to `forward_with_az_compression`
  (default `"slc"` mode) or `forward_no_az_compression` (`"rcmc"` mode).  All
  three steps (`training_step`, `validation_step`, `test_step`) now call
  `_forward_step`; quality metrics are computed in whichever domain is active.
- Added `training_mode: "slc"` parameter to `RCMCDCmodule.__init__` and to
  `configs/model/rcmc_compress.yaml`.  Override with `model.training_mode=rcmc`
  to bypass azimuth compression.

### IMPROVE 24 — Phase preservation metric at validation/test (F9)

**Files:** `src/models/components/losses.py`, `src/models/rcmc_compress_module.py`
**Change:**

- Added `phase_preservation_metric(pred, target) → (mean, std)` to
  `losses.py`.  For each pixel it computes
  $\exp(j(\phi_{\hat{s}} - \phi_s))$ on the unit circle, then measures
  $1 - |\overline{\cdot}|$ per image (0 = perfect phase preservation,
  1 = fully random phase).  Amplitude is completely factored out via
  `pred_c / (|pred_c| + ε)` before taking the mean.
- `validation_step` and `test_step` log `valid/phase_err_mean` and
  `valid/phase_err_std` (and matching `test/` variants) every epoch.

### IMPROVE 26 — Auto-generated WandB run names

**File:** `src/utils/template_utils.py`
**Change:** Added `make_wandb_run_name(cfg)` and wired it into
`early_wandb_initialization` so every run gets a human-readable name automatically.

Name pattern: `<model>-<activation>_s<seed>_L<lmbda>_<mode>_buf<buf>_<loss>_lr<lr>_b<bs>[_<N>p]`

| Token | Source | Example |
| :--- | :--- | :--- |
| `<model>` | `model._target_` → `FP` / `SHP` / fallback class name | `SHP` |
| `<activation>` | `model.net.activation` | `gdn` |
| `s<seed>` | `seed` | `s42` |
| `L<lmbda>` | `model.lmbda` | `L0.01` |
| `<mode>` | `model.training_mode` | `slc` |
| `buf<buf>` | `model.azimuth_buffer` | `buf512` |
| `<loss>` | `model.criterion._target_` → `MSE` / `Compound` / `SAR` | `SAR` |
| `lr<lr>` | `model.optimizer.lr` | `lr0.0001` |
| `b<bs>` | `data.batch_size` | `b4` |
| `_<N>p` *(optional)* | `max_products × samples_per_prod` (omitted when `spp=0`) | `_700p` |

Example: `SHP-gdn_s42_L0.01_slc_buf512_SAR_lr0.0001_b4_700p`

The name can be overridden by setting `logger.wandb.run_name` in the config.

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

`forward_with_az_compression` always uses `full_azimuth_compress_batch` (standalone
torch FFT with numpy-computed filter H).  There is no identity fallback — if metadata
or ephemeris is missing the step will raise.

---

## ⚙️ Key Config Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `data.patch_size` | `[2048, 2048]` | Core patch size (azimuth, range) |
| `data.azimuth_buffer` | `512` | Azimuth buffer lines on each side |
| `data.samples_per_prod` | `0` | Patches per product per epoch (0 = all) |
| `data.online` | `false` | Stream data from HuggingFace |
| `data.max_products_train` | `10` | Max training products |
| `model.net.nb_channels_main` | `128` | Encoder feature channels (N) |
| `model.net.nb_channels_latent` | `null` | Hyper-prior latent dim (null = 2N) |
| `model.lmbda` | `0.01` | Rate-distortion trade-off λ |

---

## 🧭 Design Notes

### Coherence loss vs phase preservation

`complex_coherence_loss` measures $|\langle\hat{s}, s^*\rangle| / (\|\hat{s}\| \cdot \|s\|)$,
which conflates phase and amplitude similarity.  A dedicated phase-only metric
measures $1 - |\overline{\exp(j(\phi_{\hat{s}} - \phi_s))}|$ per image, ignoring
amplitude entirely.  This is now implemented as `phase_preservation_metric` in
`losses.py` and logged at validation/test as `valid/phase_err_mean` and
`valid/phase_err_std` (see IMPROVE 24).  The coherence loss is kept in the training
objective as-is; the MSE term already constrains amplitude.

### `abs(y)` in the hyperprior

`z = h_a(|y|)` is the original Ballé 2018 design.  `h_s(z_hat)` outputs `scales_hat`,
which are *standard deviations* — they must be positive.  Feeding the hyperprior
`|y|` (non-negative) makes scale prediction easier.  The signed information of `y`
is preserved in the main latent path and used by `gaussian_conditional`.  This is
correct and intentional — do not change.

### `rcmc_target` in the two training modes

`_extract_from_batch` always returns `rcmc_target` (the trimmed, normalised RCMC
core).  In `"slc"` mode it is discarded.  In `"rcmc"` mode (F8),
`forward_no_az_compression` passes it as the criterion target — bypassing
`full_azimuth_compress_batch` entirely.  Select the mode via
`model.training_mode` in the Hydra config.

### H filter cache (F7)

A module-level `_FILTER_CACHE: Dict[Tuple[str, int, int, int], np.ndarray]` is used
in `sarpyx_azimuth_compression.py`.  The key is `(zfile, x_range_start, Az_total, Rg)`
— note no azimuth start, because the matched filter depends only on range geometry.

**Size estimate** (default settings `patch_az=512, buffer=1024 → Az_total=2560, Rg=512`):

$$\text{per entry} = 2560 \times 512 \times 16\ \text{bytes (complex128)} \approx 20\ \text{MB}$$

With `max_products = 6` and up to `samples_per_prod = 100` unique range positions per
product, the worst-case cache is `6 × 100 × 20 MB ≈ 12 GB` — acceptable on the
current machine (62 GB RAM, 36 GB used at peak).  The cache is an unbounded dict
(plain Python `dict`, not `lru_cache`) because `pd.DataFrame` metadata is not
hashable.  A `maxsize`-bounded LRU wrapper could be added later if memory becomes tight.

`coords_list` is now fully wired: `_extract_from_batch` returns it, and both
`forward_with_az_compression` and the `MonitorValReconstruction` callback pass it as
`coords_batch` to `full_azimuth_compress_batch`.  On the first forward pass for a given
`(zfile, x, Az, Rg)` the filter is computed and stored; all subsequent forward passes
(including across epochs) retrieve it from the cache, eliminating the `compute_azimuth_filter`
call entirely.

---

## 🔮 Future Features

| ID | Feature | Priority | Track |
| :--- | :--- | :--- | :--- |
| F11 | **Factorized Prior vs Scale Hyperprior ablation** — swap `ScaleHyperprior` for a `FactorizedPrior` via config to compare architectures | High | §Further experiments |
| F12 | **Conventional codec baselines** — JPEG, JPEG2000, WebP via CompressAI for RD-curve comparison | Low | §Further experiments |
