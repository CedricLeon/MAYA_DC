# CLAUDE.md — MAYA_DC working notes

> Orientation file for Claude/agents and the maintainer. Captures what is **true
> in the code and repo today**, including known doc-vs-reality drift. The repo
> went through a chaotic multi-machine development phase right before the EUSAR26
> deadline, so **treat the prose docs as intent, not ground truth — read the code.**
>
> Last reconstructed: 2026-06-03 (first agent pass / diagnosis).

## What this is

Neural Image Compression (NIC) applied to **Range Cell Migration Corrected
(RCMC)** Sentinel-1 SAR data — compressing *before* full focusing. The decoded
RCMC is azimuth-compressed (differentiably) into an SLC and compared to the
ground-truth SLC. Results were submitted to **EUSAR26**. Stack:
PyTorch Lightning + Hydra (from `lightning-hydra-template`) + CompressAI.

Pipeline (per `src/models/rcmc_compress_module.py`):
```
RCMC (B,2,Az+2·buf,Rg) → ScaleHyperprior (compress+reconstruct) → x̂
    → full_azimuth_compress_batch (torch FFT, differentiable) → SLC_recon
    → loss(SLC_recon, SLC_target)
```
Two training modes (`training_mode`): `slc` (default, full pipeline w/ az
compression) and `rcmc` (F8 — compare x̂ directly to RCMC, no az compression;
for products lacking ephemeris or for debugging).

## Code map (verified against source)

- `src/train.py`, `src/eval.py` — Hydra entry points.
- `src/models/rcmc_compress_module.py` — the `RCMCDCmodule` LightningModule
  (manual optimization: net optimizer + aux optimizer for the entropy
  bottleneck). Logs per-module grad norms, RD scatter to WandB.
- `src/models/components/scale_hyperprior.py` — the NIC network. Supports
  **non-square kernels** (added late; scale factor hardcoded to 3 — see
  `feat(non-square kernels)` commit, an open TODO to derive it from patch_size).
- `src/models/components/losses.py` — compound loss + metrics
  (`complex_correlation_metric`, `psnr_amplitude`, `ssim_amplitude`,
  `phase_preservation_metric`).
- `src/data/maya4_datamodule.py` — filter-based (parts/years) datamodule using
  the MAYA4 HuggingFace interface.
- `src/data/maya4_dir_datamodule.py` — **directory-based** datamodule (the
  baseline experiment uses this); scans dirs of zarr products directly,
  bypassing HF. Yields `(rcmc, slc, metadata, ephemeris, coords)` 5-tuples.
- `src/utils/sarpyx_azimuth_compression.py` — differentiable azimuth
  compression: filter `H` built in numpy (CoarseRDA maths), applied via
  `torch.fft` so gradients flow back through the decoder. Has a module-level
  `_FILTER_CACHE`.
- `src/callbacks/` — `monitor_val_reconstruction.py`, `monitor_fixed_patch.py`
  (visualization callbacks).

Config: Hydra in `configs/`. Switch loss/mode via the `training_mode` group
(`slc` | `slc_mse` | `rcmc`). Main experiment: `experiment=rcmc_compress_baseline`.

## Repo state & branch map (the important part)

| Branch | Relation | What it is |
|---|---|---|
| `main` | base `3840fe3` | bare `lightning-hydra-template` + a demo notebook. **Not the project.** |
| `dev` / `origin/dev` | 56 ahead of main, head `fbff7fd` | The real **pre-chaos** project line. Clean-ish. |
| `minimal_upgrades` | `dev` + 6 commits | **Current branch.** The 6 extra commits are the post-deadline "chaos": logger fix, caching toggles, non-square kernels, axis-inversion fix, ending in `4c38d61 "ok"`. |
| `feature/az_compr` | 7 ahead of main | Early stale spur (initial az-focusing experiments). |
| `feature/local_az_compr` | 11 ahead of main | Early stale spur (+ first Hyperprior draft). |

`4c38d61 "ok"` is a **2,928-line catch-all commit by the colleague (R. Del
Prete) pushed from the cluster** — adds the classical-codec baseline
(`scripts/evaluate_classical_codec_rd.py`), sweep/submit shell scripts,
DDP+wandb tests, the `RD-curve_plots.ipynb`, and edits across src. Its contents
were never individually reviewed by the maintainer.

Working tree (at diagnosis): `notebooks/RD-curve_plots.ipynb` modified;
untracked `Maya4/ s1isp/ srp/ output.png scripts/fix_wandb_non_square_kernels.py`.

## Vendored dependencies (decided: clone-on-install, NOT in repo)

Three packages live in the repo root as **untracked nested git clones**,
installed via `file:` refs in `pyproject.toml`:

- `Maya4/` — github.com/sirbastiano/Maya4 @ `pypi` — the dataset interface
  (`maya4` package: `GT_MIN/MAX`, `RC_MIN/MAX`, `minmax_normalize/inverse`).
- `srp/` — github.com/sirbastiano/srp @ `main` — **sarpyx**, SAR processing
  (CoarseRDA). Source of the azimuth-compression maths.
- `s1isp/` — github.com/avalentino/s1isp @ `main` — Sentinel-1 ISP (L0) decoder.
  Has a stray untracked `_huffman.c`. **Not mentioned in install docs** (drift).

**Key finding (2026-06-03):** the local clones have **no local modifications** —
Maya4 is 20 commits *behind* `origin/pypi`, srp 70 *behind* `origin/main`,
s1isp *even* with upstream; none have local-only commits. The stray
`s1isp/_huffman.c` is a generated Cython artifact. So whatever was once modified
is already upstreamed → **de-vendoring loses nothing.** `pyproject.toml`
currently pins `maya4 @ file:Maya4` and `sarpyx @ file:srp` (so a fresh clone
can't `pip install` without the dirs present); `s1isp` is not a declared dep.
Target: replace `file:` refs with installable refs (PyPI or `git+https` commit
pins) so a fresh MAYA_DC clone installs deps and runs without manual cloning.

## data/ (gitignored: `/data/` in .gitignore)

- **~144 GB of DEPRECATED local data** — `PT1/ PT2/ PT4/ TEST/
  test_complete_download/ sentinel1_copernicus/`. Canonical data now lives on
  **HuggingFace buckets**; these local copies are stale. Safe to consider for
  deletion (confirm before destructive action).
- `data/from_cluster/` (~3 GB) — **paper artifacts hand-ported from the
  cluster** (results, figures, and the scripts that made them). Has heavy
  zip-vs-extracted duplication. See the diagnosis report / cleanup plan. The
  paper experiment = `sq` vs `nsq` kernels × λ∈{3,10,15,20,50,1000} × seed0,
  benchmarked vs JPEG / JPEG2000 / WebP. Canonical RD notebook is the
  **from_cluster** copy (not the tracked `notebooks/` one).

`logs/` is a **dangling symlink** to a cluster path
(`/lustre/scratch/.../maya_dc_outputs`) — not accessible locally.

## Known doc-vs-reality drift (docs are stale; trust code)

- README + QUICKSTART: "`Maya4/` and `srp/` are NOT in this repo" — they *are*
  present (untracked); `s1isp/` unmentioned.
- QUICKSTART layout + `progress_tracking.md` reference
  `scripts/sarpyx_azimuth_compression.py` and `scripts/download_maya4_data.py` —
  **both deleted** (commit `cdd3bf6`).
- λ defaults disagree everywhere (README/QUICKSTART say 1.0 / 0.01 / 100) vs the
  baseline config (`model.criterion.lmbda: 1000`).
- `configs/experiment/rcmc_compress_baseline.yaml` `train/val/test_dir` point at
  `/lustre/...` cluster paths (won't resolve locally).

Existing prose docs (read for *intent*, verify before trusting):
`README.md`, `docs/QUICKSTART.md`, `IMPLEMENTATION_SUMMARY.md` (bug/feature
history F1–F15), `CONVERGENCE_ANALYSIS.md`, `progress_tracking.md`,
`.github/copilot-instructions.md`.

## Commands

```bash
conda activate MAYA_DC                                        # env from environment.yaml → pyproject.toml
python src/train.py experiment=rcmc_compress_baseline         # full run
python src/train.py experiment=rcmc_compress_baseline +trainer.fast_dev_run=2   # smoke test
python src/train.py debug=fdr        # fast-dev-run preset
python src/train.py debug=overfit    # overfit 1 batch
```
Constraint: `patch_size[0] + 2*azimuth_buffer` must be divisible by 16
(datamodule asserts on startup).

## Cluster context

The cluster used **PBS / OpenPBS** (not SLURM): `#PBS` directives, `qsub`/`qstat`,
queues `gpu4_std` / `cpu_std`, project root `/lustre/projects/1001/rdelprete/...`.
All `main-*.sh`, `wandb-sync.sh`, and `scripts/submit_classical_codec_baseline.sh`
are PBS launchers hardcoding those paths → delete on cleanup (knowledge preserved
in docs + the `cluster-chaos` tag). The experiment sweep = λ∈{1…1000} × seeds,
defined by `main-sweep.sh` + `main-sweep-run.sh` + `configs/hparams_search/rcmc_grid.yaml`.

## Open decisions / cleanup status (as of 2026-06-03)

Diagnosis done. **Safety tag `cluster-chaos` created** (annotated, local-only,
→ `4c38d61`) so nothing is lost.

**Execution progress (2026-06-11), branch `clean-chaos`:** P0 branch ✓ · P1 de-vendor ✓
(deps now install from git/PyPI; `Maya4/ srp/ s1isp/` removed — see
`docs/dependency-bump-notes.md`) · P2 cluster glue removed ✓ (PBS launchers + `logs`
symlink deleted; commands preserved in `docs/cluster.md`). The pre-cleanup descriptions
above (vendored clones, `logs` symlink, etc.) are reconciled in the P6 doc truth-pass.

**Locked decisions (2026-06-11):**
- Work branch `clean-chaos` off `minimal_upgrades` → merge to `dev` → (after docs) release `main`.
- De-vendor: `Maya4/ srp/ s1isp/` are stale clones w/ no local mods. Replace
  `file:` refs with **PyPI** deps (`maya4` 0.1.2, `sarpyx` 0.1.10 both published);
  record last-good commits (Maya4 `dc798a3`, srp `4024da1`) in a comment. **Drop
  `s1isp`** (only the demo notebook used it; not on PyPI). Verify w/ clean-venv install + smoke test; fall back to an older pinned version only if API drift breaks it.
- Delete cluster glue (PBS launchers `main-*.sh`, `wandb-sync.sh`,
  `scripts/submit_classical_codec_baseline.sh`, `logs` symlink) AFTER distilling
  their commands into `docs/` (ESA SpaceHPC / PBS, concise).
- Analysis artifacts all go to `notebooks/` (RD-curve notebook = from_cluster
  canonical, `plot_combined_rd_curves.py`, `make_overview_figure.py`). **Recon
  viewer set deleted** (`viewer_helpers.py`, `reconstruction_viewer.ipynb`,
  `_build_reconstruction_viewer.py`) — overview script only. Also delete
  `notebooks/run_rd_curve_plots.py` (headless executor) + `RD-curve_plots.png`.
- Keep `scripts/experiment_block_processing.py` (add header: studies azimuth
  buffer-size effect). Delete `scripts/fix_wandb_non_square_kernels.py` (one-off).
- Raw `.npy` results stay in `data/` (gitignored, off the git repo, on disk);
  scripts needing them get an explicit note; `docs/` gets a concise raw-data layout doc.
- Delete ~144 GB deprecated `data/PT*/TEST/...` (disk only) — **pending final OK**.
- Docs/README drift fixed LAST (incl. λ∈{1…1000}).

(superseded planning context below)
Was awaiting maintainer decision on repo strategy (e.g., stale-branch the cluster chaos vs. release a clean state to
`main`). When acting:
- Decide fate of the 5 branches (consolidate onto one clean line).
- Dedupe `data/from_cluster/` (drop zips that have extracted copies; archive the
  1.5 GB `results_extracted/` raw reconstructions off-repo; promote figure
  scripts + final figures into `scripts/`/`docs/`).
- Fix the doc drift listed above once the clean structure is settled.
