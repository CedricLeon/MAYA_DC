# Dependency de-vendoring — bump notes (P1)

Scratch log for the `file:` → PyPI dependency migration (plan phase P1). Records the
pre-bump baseline, what broke on the bump, how each breakage was classified/fixed, and
the maintainer decisions taken. Keep until P1 is accepted; then distil the durable bits
into `CLAUDE.md` / README and delete this file.

## Package version landscape (verified 2026-06-11)

| Package | Local clone (old `file:` dep) | PyPI available | Bump direction |
|---|---|---|---|
| `maya4`  | **1.0.0** @ `dc798a3` (branch `pypi`) | `0.1.0, 0.1.1, 0.1.2` (latest **0.1.2**) | **downgrade** (1.0.0 → 0.1.2) — high API-gap risk |
| `sarpyx` | **0.1.5** @ `4024da1` (branch `main`) | …`0.1.8, 0.1.9, 0.1.10` (latest **0.1.10**) | upgrade (0.1.5 → 0.1.10) |
| `s1isp`  | even w/ upstream | not on PyPI | **dropped** (demo-only) |

Note the version *numbers* are misleading: the local maya4 declares `1.0.0` but the
published PyPI lineage tops out at `0.1.2`. PyPI is therefore an older codebase than the
local clone for maya4. Pin chosen: `maya4>=0.1.0` (robust regardless of which 0.1.x is
newest), `sarpyx>=0.1.10`.

## Pre-bump baseline (env `MAYA_DC`, old `file:` deps: maya4 1.0.0 / sarpyx 0.1.5)

1. **Import check** — all OK: `maya4`, `sarpyx`, `CoarseRDA`, `ProductHandler`,
   `range_dec_to_sample_rate`, constants, `KPatchSampler/SampleFilter/SARTransform/SARZarrDataset`,
   `NormalizationModule`, `GT/RC` constants, `minmax_*`.
2. **pytest** — **9 passed, 11 failed**. The 11 failures are **pre-existing stale template
   tests**, NOT dependency-related: lightning-hydra-template MNIST tests
   (`test_train.py`, `test_eval.py`, `test_sweeps.py`) fail because the project's
   `MonitorValReconstruction` callback rejects the template `MNISTLitModule`
   (`ValueError: MonitorValReconstruction expects RCMCDCmodule, got 'MNISTLitModule'`),
   plus template sweep/ddp-sim issues. **This is the baseline pass/fail set** — post-bump,
   only *new* failures count as regressions.
3. **Numerical gate** — `scripts/validate_azimuth_pipeline.py` on
   `data/.../PT4/...05d521.zarr`, core patch az=[3000:3512] rg=[12000:12512], buffer 512:

   | Metric | CoarseRDA vs GT | Custom-FFT vs GT | Custom vs CoarseRDA |
   |---|---|---|---|
   | Complex corr | 0.999986 | 0.957260 | **0.957214** |
   | Magnitude corr | 0.999972 | 0.919209 | 0.919124 |
   | PSNR [dB] | 64.49 | 29.81 | 29.79 |
   | SSIM | 0.999968 | 0.913190 | 0.913056 |

   **Invariant to preserve post-bump:** Custom-vs-CoarseRDA complex corr ≈ **0.957**
   (NOT ≈1.0 — the custom FFT was never a perfect match), and CoarseRDA-vs-GT ≈ 0.99999.
   A drop here = sarpyx changed CoarseRDA internals → Class C escalation.
4. **Smoke train** (`fast_dev_run=2`, local clean PT4 products, single GPU, logger=csv) —
   **ALREADY BROKEN in the baseline** with `AttributeError: module 'zarr' has no attribute
   'hierarchy'` from `Maya4/maya4/dataloader.py:747` (`isinstance(store, zarr.hierarchy.Group)`).
   Cause: `pyproject` pins `zarr>=3.1.5`; zarr v3 removed `zarr.hierarchy` (v2 API), but local
   maya4 1.0.0 still uses it. So the full training datamodule does not run in this env
   regardless of the bump. The maya4 bump may fix this (if PyPI maya4 supports zarr v3) or not.

## Post-bump results

### Step A — naive PyPI bump (`maya4>=0.1.0` → 0.1.2, `sarpyx>=0.1.10` → 0.1.10)
1. **Import ladder: PASS** — all maya4 + sarpyx symbols resolve.
2. **pytest: 9 passed / 11 failed — IDENTICAL set to baseline.** No new failures → no dep regression.
3. **Numerical gate: bit-identical to baseline** (Custom-vs-CoarseRDA complex corr 0.957214,
   CoarseRDA-vs-GT 0.999986, all metrics unchanged). sarpyx 0.1.5→0.1.10 changed nothing
   numerically. **No Class C drift.** ✅
4. **Smoke train: STILL BROKEN** — `AttributeError: module 'zarr' has no attribute 'hierarchy'`
   from `site-packages/maya4/dataloader.py:736`. **PyPI maya4 0.1.2 declares `zarr>=3.1.5` yet
   its code uses the v2-only `zarr.hierarchy.Group`** → cannot open products with the project's
   required zarr v3. Same failure as baseline (Class B, upstream maya4 bug).

### Step B — root-causing maya4 (the blocker)
- Upstream `origin/pypi` HEAD (`c535776`, ~20 commits ahead of the vendored `dc798a3`) **fixed it**:
  `isinstance(store, zarr.Group)` (v3-correct), **0** `zarr.hierarchy` usages. But `c535776` is
  **not published to PyPI** (PyPI tops at the obsolete 0.1.2).
- The newer maya4 is also **HF-bucket-aware**: `api.py` imports `download_bucket_files,
  list_bucket_tree` from `huggingface_hub` — exactly the API for the new data home
  `huggingface.co/buckets/ESA-philab/Maya4`. It declares `huggingface-hub>=1.5.0`
  (installed was 1.3.2; those functions appear in hub ≥1.5).
- ⇒ **PyPI maya4 0.1.2 is doubly obsolete**: zarr-v2 code AND no bucket support. It is a dead end
  for this project.

### Step C — VERIFIED working recipe (installed + smoke-tested end-to-end)
- `maya4 @ git+https://github.com/sirbastiano/Maya4@pypi` (built from `c535776`; v3-clean, bucket-aware)
- `sarpyx>=0.1.10` (PyPI; numerically identical, e2e-clean)
- `huggingface-hub>=1.5.0` (installed 1.19.0; provides the bucket API)
- **Smoke train PASSED** end-to-end (train→val→test, `fast_dev_run=2`, local PT4 products):
  `test/complex_corr_mean 0.985`, `psnr_amp 15.0`, `ssim_amp 0.357`, `rate 0.169`.
- Only the `MonitorFixedPatch` callback fails — unrelated to deps: it is hardcoded to a cluster
  product via the repo-root `s1c-...0040d4.json` (`/lustre/.../PT1/...zarr`). Cleanup item, not P1.

### Latent config bug found (note for later)
`configs/experiment/rcmc_compress_baseline.yaml` overrides `callbacks.monitor_fix_patch` but the real
key is `callbacks.monitor_fixed_patch` — the typo'd override is a silent no-op (its `verbose: True`
never applies).

### DECISION 1 (RESOLVED) — maya4 source
Maintainer chose: **maya4 @ git `pypi` branch** (PyPI release is unusable). Maintainer will ask the
upstream owner to publish to PyPI; pyproject carries a `TODO(maya4-pypi)` to switch then. sarpyx
stays PyPI 0.1.10. `huggingface-hub` pin bumped `>=1.0.0` → `>=1.5.0`. ✅ applied.

### Other pre-existing install blockers found while validating `pip install -e .`
(Neither caused by de-vendoring; both independently break a fresh-clone install — in P1 scope.)

**(a) Invalid build backend — FIXED (Class A).** `build-backend = "setuptools.backends.legacy:build"`
(no such module; introduced in `fbff7fd`) → `pip install` couldn't even build the project's metadata.
Corrected to `setuptools.build_meta`. (`maya-dc` was never actually pip-installed before — the repo
ran via `rootutils` pathing + editable maya4/sarpyx, so the broken backend went unnoticed.)

**(b) compressai vs numpy — DECISION 2 (pending).** `pip install -e .` fails to resolve:
`compressai` latest on PyPI is **1.2.8**, which hard-caps **`numpy<2.0`**, while our pyproject pins
`numpy>=2.0.0`. The env nonetheless *runs* fine (compressai 1.2.8 + numpy 2.3.5; smoke train passed),
so the cap is over-conservative metadata. Facts:
- compressai **master** (`1.2.9.dev0`, unreleased) already relaxed it to `numpy>=1.24.4`.
- Only *our own* pin forces numpy≥2; every other dep is fine on numpy≥1.26 (zarr≥1.26, numcodecs≥1.24,
  scikit-image≥1.24, sarpyx any, torch none).
Options put to maintainer:
- **A:** `compressai @ git+.../CompressAI` (master; keeps tested numpy 2) — 2nd git dep + source build.
- **B:** relax our pin to `numpy>=1.26,<2`, keep PyPI `compressai` wheel — moves off the tested numpy 2.

**RESOLVED → A.** Maintainer steered to keep `numpy>=2.0` (they'd hit this before; the compressai cap is
spurious and removed on main). PyPI re-checked: latest is still 1.2.8 *with* the `numpy<2` cap, so the
fix is git-only. Applied: `compressai @ git+https://github.com/InterDigitalInc/CompressAI` (builds
1.2.9.dev0). Also fixed the build backend (a). README/QUICKSTART install sections updated (no more
"clone Maya4/srp"; note the git deps + C-compiler need). `TODO(compressai-pypi)` in pyproject to switch
back once 1.2.9 releases.

## FINAL verified stack (P1 done)
`maya4 @ git+…/Maya4@pypi` (c535776) · `compressai @ git+…/CompressAI` (1.2.9.dev0) ·
`sarpyx>=0.1.10` (PyPI) · `huggingface-hub>=1.5.0` · `numpy>=2.0.0` (kept) · zarr 3.1.5.

Verification ladder (env `MAYA_DC`, post-fix):
1. `pip install -e ".[dev]" --dry-run` → **resolves cleanly** (`Would install maya-dc-0.1.0`, exit 0).
   Headline "fresh clone installs without manual cloning" goal met (resolver-level).
2. Import ladder → **PASS** (all maya4 + sarpyx symbols).
3. pytest → **9 passed / 11 failed = identical to baseline** (the 11 are pre-existing stale template
   tests; tracked as a separate cleanup, not a dep regression).
4. Numerical gate → **bit-identical to baseline** (Custom-vs-CoarseRDA 0.957214, etc.). No drift.
5. Smoke train (`fast_dev_run=2`, local PT4, GPU) → **PASSES end-to-end** (`test/complex_corr 0.985`,
   `psnr_amp 15.0`, `ssim 0.357`, `rate 0.169`). NB this is a net improvement: the pipeline was
   *broken at baseline* (zarr.hierarchy); de-vendoring fixed it.

Caveats / follow-ups carried forward:
- Fresh-venv (from-scratch) install not yet run — covered by the plan's end-to-end check; resolver
  dry-run + in-place run are strong proxies.
- `MonitorFixedPatch` callback is hardcoded to a cluster product via repo-root
  `s1c-…0040d4.json` (`/lustre/.../PT1`) → fails unless that product is present (disabled in smoke).
  Cleanup item for a later phase.
- Latent config typo: experiment overrides `callbacks.monitor_fix_patch` but the real key is
  `monitor_fixed_patch` (silent no-op). Fix in a later phase.
- 11 stale template tests (MNIST) should be removed/adapted for a clean release.
