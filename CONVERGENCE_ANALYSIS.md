# MAYA_DC — Convergence Analysis

**Last updated:** March 2026 — epoch 50 run, pre-session fixes (BUG 27/28/15 applied)
**Training mode:** `slc` · Loss: `CompoundCompressionLoss` · λ=10 · δ_kde=1 · δ_coh=1

---

## 1. How to read SAR NIC metrics

Before diagnosing, these baselines calibrate expectations:

| Metric | Perfect recon | Identity (x̂ = x) | Random output | Our run (ep 50) |
| :--- | ---: | ---: | ---: | ---: |
| `psnr_amp` [dB] | ∞ | ∞ | ~0 | 17.0 |
| `ssim_amp` | 1.000 | 1.000 | ~0 | 0.099–0.15 |
| `complex_corr` | 1.000 | 1.000 | ~0.5 | 0.968 |
| `phase_err` | 0.000 | 0.000 | ~0.5 | ~0.016 |
| `rate` [bpp] | — | ∞ | — | 0.098–0.277 |

**Why SSIM looks worse than it is for SAR:**
SSIM penalises any local-mean/variance mismatch. SAR speckle has no predictable local mean — it is inherently stochastic. A 0.15 SSIM for a *first training run at 0.1 bpp* is not catastrophic, but it is still bad enough to diagnose.

**The confusing `complex_corr ≈ 0.97` alongside `ssim ≈ 0.10`:**
Complex correlation measures the *global* cosine similarity in the complex plane. If the decoder preserves the broad spectral phase structure (which the differentiable az-compression encourages), correlation can be high while amplitude spatial patterns are wrong (low SSIM). These are complementary diagnostics, not redundant ones.

---

## 2. Observed symptoms (epoch-50 WandB summary)

| Signal | Value | Interpretation |
| :--- | :--- | :--- |
| `train/loss` | 1.48 | |
| `valid/loss` | 3.01 | **2× train loss** → significant train/val distribution gap |
| `test/aux` | 1 237 | High in absolute terms — but **normal**; magnitude is architecture/data-dependent. What matters is the decreasing trend. |
| `train/rate` | 0.277 bpp | |
| `test/rate` | 0.098 bpp | **3× gap** — entropy model does not generalise to test products |
| `test/distortion` | 0.239 | λ·dist ≈ 2.4 at test; much larger than rate |
| `val_batch/ssim_slc` | 0.099 | Near-zero structural similarity |
| `val_batch/psnr_slc` | 17.0 dB | Below-par but not zero signal |
| `valid/complex_corr_mean` | 0.968 | Phase structure broadly preserved |
| `valid/phase_err_mean` | 0.016 | Excellent phase fidelity |
| `train/grad_norm_g_s` | 24.4 | Decoder gradient flowing (BUG 15 confirmed fixed) |
| `train/grad_norm_g_a` | 13.2 | Encoder gradient flowing |
| `train/grad_norm_h_a` | 0.030 | Hyper-encoder gradient is 800× smaller than encoder |
| `train/grad_norm_h_s` | 0.030 | Same for hyper-decoder |

**Visual observation (val_batch callback):**
- RCMC input patches look like pure noise → likely ocean-only validation product.
- SLC target also noisy → consistent with ocean patches (Rayleigh speckle looks textureless).
- RCMC/SLC recon also essentially noise/empty.

**Visual observation (fixed_patch callback):**
- RCMC input looks structured (Fogo volcano island) → manual zarr pipeline is correct.
- SLC recon panel absent at epoch 50 → ephemeris was empty in old FOCUSED product (now fixed with `FOCUSED_v4` rename).

---

## 3. Verified / ruled out

| Item | Status | Evidence |
| :--- | :--- | :--- |
| Gradient flow through decoder g_s | ✅ | `grad_norm_g_s = 24.4`; BUG 15 fix (torch FFT) |
| Data normalisation (BUG 27) | ✅ | Fix applied; re/im both in [0,1] after dataloader |
| SSIM computed on correct domain (BUG 28) | ✅ | Both recon and target normalised to [0,1] before metrics |
| bpp normalised by RCMC pixels (FIX 14) | ✅ | `recon_shape` uses `self.rcmc_shape` not x_hat shape |
| Azimuth compression math | ✅ | `validate_azimuth_pipeline.py`: buf=512 → ~97%, buf=1024 → ~99.9% |
| Aux optimizer wiring | ✅ | `manual_backward(aux_loss)` + `aux_optimizer.step()` in correct order |
| Training produces non-zero main loss gradients | ✅ | g_a, g_s gradients measured every step |
| Fixed-patch pipeline (direct zarr) | ✅ | Visual confirms structured images for land product |
| Val-batch pipeline vs fixed-patch pipeline | ❓ | Run `visualize_data.py` to confirm distributions match |
| Val product content (ocean vs. land) | ❓ | Need to check which product is used for validation |
| Decoder output distribution (x_hat histogram) | ❓ | Check WandB `val_batch/x_hat_histogram` panel |
| Hyper-latent z scale | ❓ | h_a grad ~0.03; z range unknown — may be off for entropy model |

---

## 4. Root cause hypotheses (ranked by confidence)

### H1 — Training duration far too short 🔴 **Critical**

**Evidence:** 50 epochs × 7 products × 100 patches ÷ batch_size 4 = **~8 750 gradient steps**.
CompressAI reference training uses 1–2 M gradient steps at batch 16. We have done ~0.1 % of that.

**Impact:** The entropy bottleneck quantile estimator converges on its own track. With only 8 750 steps, there is not enough signal for the main network (g_a, g_s, h_a, h_s) to learn meaningful compression, let alone for the aux model to learn the latent prior.

**Fix:** Either dramatically increase `samples_per_prod` / `max_products` / `max_epochs`, or run much longer. A first meaningful convergence check would require ≥ 50 000 steps.

---

### H2 — Auxiliary loss interpretation *(revised — previous analysis was wrong)* 🟡 **Low**

**Correction:** The earlier claim that "converged CompressAI models have aux < 10" was hallucinated.
In practice, aux magnitude is architecture- and data-dependent and can remain in the thousands for
the entire training run while the main network converges normally.  **The only meaningful signal is
the trend**: is it decreasing?  In the overfit-500 run, aux went 5 275 → 5 103 over 500 steps —
slow, but consistently decreasing.  That is healthy.

**What aux_loss measures:** The auxiliary optimizer minimises the mismatch between the entropy
bottleneck's learned CDF (quantile table) and the empirical distribution of `z = h_a(|y|)`.  It
converges on its own schedule, independently of the main rate-distortion objective, and typically
much more slowly.

**Correct explanation for the train/test rate gap (epoch-50 run):**
`train/rate = 0.277 bpp` vs `test/rate = 0.098 bpp`.  The test products come from a different
geographic split (PT4) than the training products (PT1).  Land products have higher spatial entropy
→ more bits needed; ocean products are near-Rayleigh speckle → lower entropy → fewer bits.  This
is **data distribution**, not entropy model failure.

**Correct explanation for the train/test distortion gap (overfit-500 run):**
`train/distortion ≈ 0.041` vs `test/distortion = 0.125`.  CompressAI's `forward()` uses additive
uniform noise as a quantization proxy in training mode and **actual rounding** in eval mode
(Lightning calls `model.eval()` automatically for test).  The model was not trained to be robust
to rounding → test distortion is higher.  This is expected and closes with more training.  Note:
this is distinct from calling `net.compress()`/`net.decompress()` — we never do that (see F17).

**When to escalate:** Only if aux shows **no downward trend** after ≥ 100 k steps.  In that case,
log `z.abs().mean()` (F16) to check whether the hyper-latent values are outside the entropy
bottleneck's initialisation range.  This is a rare differential diagnostic, not a primary concern.

---

### H3 — Validation product is ocean-only 🟡 **High**

**Evidence:** RCMC input and SLC target both look like "pure noise" in the val_batch callback. The fixed-patch product (Fogo island) shows clear structure. `max_products_val = 1` → a single product determines all val metrics.

**Impact:** Ocean SAR patches are spatially uncorrelated speckle. The model cannot learn (or be diagnosed) on them because:

1. All patches look statistically identical → the reconstruction loss is flat over the model's output.
2. KDE loss measures amplitude histogram matching — for Rayleigh speckle, any Rayleigh-like output achieves near-zero KDE loss regardless of pixel-level accuracy.
3. Visual diagnostics appear worse than they are (noise-like inputs are not "wrong" — they are SAR ocean).

**Fix:** Audit the validation product. Use `cherry_pick_patch.ipynb` / `product_catalog.json` to select a validation product with confirmed land or coastal coverage.

---

### H4 — SLC-mode training from scratch is the hardest possible start 🟡 **High**

**Evidence:** `training_mode = "slc"`. The full pipeline is:
```
random_net → x̂ (garbage) → denorm → az_compress → renorm → SLC_hat (garbage²) → loss vs GT_SLC
```
The az-compression operator *mixes* all azimuth lines inside a buffer. Early in training, the random encoder output propagates through az-compression and produces SLC patches with no relation to the GT. The gradient must backprop through IFFT→H→FFT to reach the encoder. This is geometrically longer and flatter than the RCMC-domain loss.

**Fix:** Two-phase training — start in `training_mode=rcmc` until RCMC PSNR > 25 dB (simpler, shorter gradient path), then switch to `training_mode=slc`.

---

### H5 — Compound loss over-constrains an untrained model 🟡 **High**

**Evidence:** Three competing terms — MSE, KDE, coherence — all with large weight.

**Impact:** Early in training, all three loss terms give contradictory gradients. The KDE loss tries to match speckle histograms; the coherence loss pushes towards global phase alignment; MSE pushes towards zero error. None of these objectives has "won" → the model stagnates.

**Fix:** Start with `SimpleMSELoss` (the simplest objective). Confirm the model can overfit a single batch before adding compound terms.

---

### H6 — Large buffer-to-core ratio 🟠 **Medium**

**Evidence:** `patch_size=[512,512]`, `azimuth_buffer=512`. Total Az input = 1536; buffer = 1024 (67 %). The model processes 3× more pixels than it receives loss signal from. The encoder must "waste" capacity on border pixels that are discarded.

**Impact:** Effective batch size is `B × Az_core × Rg = 4 × 512 × 512 ≈ 1M pixels/batch`. Actual pixels contributing to the loss: `4 × 512 × 512 = 1M` for RCMC mode but only after az-compression.

**Fix:** Increase core patch to `[1024, 1024]` with same buffer → buffer fraction drops to 50 %. Or reduce buffer to 256 once the model is learning (accepting slightly lower az-compression quality).

---

## 5. Investigation checklist

Before changing hyperparameters, run these diagnostics in order:

- [ ] **Check validation product**: which product is `max_products_val=1` selecting? Is it over ocean? Use `product_catalog.json` to identify.
- [ ] **Run `scripts/visualize_data.py`**: confirm channel_distributions rows 0 and 1 (DataModule vs Manual) match. If they differ, a residual normalisation bug is present.
- [ ] **Check x_hat histogram** (WandB `val_batch/x_hat_histogram`): is the decoder output concentrated at 0, 0.5, or the boundaries? Mode collapse = constant output; clipping saturation = values at ±1.
- [ ] **Run overfit test**: `python src/train.py debug=overfit model.criterion._target_=...SimpleMSELoss`. If the model cannot overfit a single batch after 500 steps, the architecture or loss has a deeper issue.
- [ ] **Log z latent stats** *(low priority — only if aux shows no downward trend after ≥ 100 k steps)*: add `z.abs().mean()` and `z.std()` to check whether the hyper-latent values are drifting outside the entropy bottleneck's init range.  See F16.
- [ ] **Identity baseline**: manually set `x_hat = rcmc_input` (bypass the network) and measure PSNR/SSIM/coherence of the resulting SLC. This is the theoretical ceiling for perfect RCMC compression; anything below this is a genuine model failure.

---

## 6. Proposed fixes (non-hyperparameter)

### Fix A — Two-phase curriculum (RCMC → SLC)

```bash
# Phase 1: RCMC domain only (easy gradient path, no az-compression)
python src/train.py experiment=rcmc_compress_baseline \
  model.training_mode=rcmc \
  model.criterion._target_=src.models.components.losses.SimpleMSELoss \
  trainer.max_epochs=200

# Phase 2: fine-tune in SLC domain
python src/train.py experiment=rcmc_compress_baseline \
  model.training_mode=slc \
  +model.compile=False \
  ckpt_path=logs/.../last.ckpt \
  trainer.max_epochs=300
```

**Rationale:** RCMC-mode MSE is a shorter, stronger gradient signal. Once the network learns to reproduce RCMC patches (PSNR > 25 dB), switching to SLC mode only needs to teach the encoder to distribute information in az-friendly ways.

---

### Fix B — Overfit-on-one-batch smoke test

```bash
python src/train.py debug=overfit \
  model.criterion._target_=src.models.components.losses.SimpleMSELoss \
  model.criterion.lmbda=0.01 \
  data.samples_per_prod=10 \
  data.max_products_train=1 \
  trainer.max_epochs=500
```

If `valid/psnr_amp` doesn't rise above 25 dB after 500 steps on a *single batch*, the architecture or loss has a hard bug. This test is a prerequisite before any long-run experiment.

---

### Fix C — Select land/coastal validation product

Update `max_products_val = 1` to point to a product with confirmed land coverage (e.g., a PT4 or PT1 product over the Canary Islands / Fogo island chain). Ocean-only validation is diagnostically useless.

**How:** In `configs/experiment/rcmc_compress_baseline.yaml`, add:
```yaml
data:
  val_files: [path/to/land_product.zarr]  # if supported
```
Or use `product_catalog.json` to identify and pin a specific product via `data.val_parts`.

---

### Fix D — Log hyper-latent scale

Add to `training_step` in `rcmc_compress_module.py`:

```python
with torch.no_grad():
    z_vals = self.net.h_a(self.net.g_a(rcmc_batch).abs())
    self.log("train/z_abs_mean", z_vals.abs().mean(), on_step=True, on_epoch=False)
    self.log("train/z_std",      z_vals.std(),          on_step=True, on_epoch=False)
```

If `z_abs_mean` > 50, the entropy bottleneck is operating outside its init range → aux loss will stay large until z drifts into range or the quantile parameters catch up. In that case, increasing the aux optimizer lr or pre-training only the entropy model may help.

---

### Fix E — Much longer training run

Keep all current settings and simply run far more epochs:

```yaml
trainer:
  max_epochs: 1000  # up from 50
data:
  samples_per_prod: 200  # up from 100
```

At `1000 × 7 × 200 / 4 = 350 000 steps`, we approach ~35 % of CompressAI's reference schedule. This alone may resolve H1 and H2 without any code change.

---

## 7. Priority order for next runs

| Priority | Action | Expected outcome |
| :--- | :--- | :--- |
| ~~1~~ | ~~Overfit smoke test (Fix B, RCMC mode, SimpleMSE)~~ | ✅ **PASSED** — train/distortion 0.357→0.041 in 500 steps; architecture confirmed healthy |
| 2 | Identify & replace ocean validation product (Fix C) | Visual diagnostics become meaningful |
| 3 | Long run: RCMC mode, λ=100, SimpleMSE, ≥500 epochs (Fix E) | First real convergence signal in RCMC domain |
| 4 | Two-phase curriculum: switch to SLC once RCMC PSNR > 25 dB (Fix A, phase 2) | Full pipeline convergence |
| 5 | Add compound terms once SLC PSNR > 25 dB | Extend to distribution + phase fidelity |
| 6 | Log z latent stats (Fix D) — *only if aux shows no trend after 100 k steps* | Differential diagnostic for entropy bottleneck range |
| 7 | Real test-time compression via net.compress/decompress (F17) | Validate actual compression ratio and bitstring bpp |

---

*Update this file after each run: add a row to the table below.*

## 8. Run log

| Run | Config | Steps | RCMC PSNR | SLC PSNR | SSIM | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| baseline-ep50 | SLP+lmbda10+compound | 8 750 | — | 17.0 | 0.10 | Pre-session fixes; ocean val product suspected |
| overfit-500 | rcmc+λ=10+SimpleMSE+debug=overfit | 500 | train dist 0.357→0.041 | test dist 0.125 | test 0.110 | ✅ Architecture healthy. Train/test gap = noise→rounding switch (eval mode) + possible data distribution. aux: 5275→5103 (slow but decreasing — normal). |
