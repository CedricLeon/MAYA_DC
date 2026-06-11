# Cleanup summary — `clean-chaos` branch (TEMPORARY)

> Scratch record of the release-prep cleanup. **Delete this file before merging to `dev`.**
> Detailed dependency log: `docs/dependency-bump-notes.md`. Plan: `.claude/plans/…corbato.md`.

## Surprises vs. the plan (and how they were handled)

1. **maya4 PyPI unusable** — 0.1.2 still uses zarr-v2 `zarr.hierarchy` (breaks on our zarr 3) and
   lacks HF-bucket support → used `maya4 @ git pypi` branch (TODO to switch to PyPI when published).
2. **Pipeline was already broken at baseline** — pre-bump smoke failed on `zarr.hierarchy` too;
   the maya4-git bump *fixed* it (end-to-end smoke now passes).
3. **compressai ⟷ numpy** — PyPI compressai 1.2.8 caps `numpy<2`; cap removed only on main →
   used `compressai @ git main`, kept `numpy>=2`.
4. **huggingface-hub** — bucket-aware maya4 needs `>=1.5`; bumped the pin.
5. **Invalid build backend** (pre-existing) — `setuptools.backends.legacy:build` → `setuptools.build_meta`;
   the repo had never actually been `pip install`-able (ran via rootutils pathing).
6. **Deprecated data broke smoke** — `data/PT1/.cache/…` incomplete zarrs crash the scan; used a clean
   mini-dir for testing (the bad cache is part of the P7 delete).
7. **Plots needed per-patch CSVs** — promoted them too (incl. a 7 MB jpeg one); both plot tools verified.
8. **Pre-commit gates** — scoped `interrogate` to `src/` (analysis code excluded); fixed an `F541`.
9. **Bucket filter mismatch** — default config filters (`PT2`,`hh/hv`) match none of the bucket's
   `vv`/2025 products; documented (online streaming verified with widened filters).

## Fixes applied (B1–B6)

- **B1/B4 — MonitorFixedPatch:** kept crash-loud; removed it from the default callback list (opt-in now);
  relocated the patch JSON repo-root → `configs/fixed_patches/`, pointed the product field at the bucket
  name + a `_comment`, updated config + the callback's search path. (The 16 GB product can't be vendored;
  re-enable with a local copy or your own product.)
- **B2** — removed the `monitor_fix_patch` typo block from the experiment (was a silent no-op).
- **B3** — deleted the 11 stale lightning-template MNIST tests (`test_train/eval/sweeps.py`).
- **B5** — experiment `train/val/test_dir`: dead `/lustre/...` → `${paths.data_dir}/maya4/{train,val,test}` + comment.
- **B6** — deleted stale gitignored old-workflow files in `data/` (product catalogs, split lists, fogo patch json).

## Still open (flagged, not done)
- Broader MNIST/template de-crufting: `src/{models/mnist_module,data/mnist_datamodule,models/components/simple_dense_net}.py`,
  `configs/{model,data}/mnist.yaml`, `configs/experiment/example.yaml`, `tests/test_datamodules.py`,
  and the fact that the **default** `train.yaml` still composes MNIST (project runs via `experiment=…`). Decide for release.
- Doc truth-pass (B7) — pending validation.
