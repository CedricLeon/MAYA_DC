# Off-git raw data (`data/from_cluster/`)

`data/` is gitignored, so the large paper artifacts below live **on local disk only** —
they are not in the repo and not recoverable from git. They are regenerable (re-run
training / codec eval), so treat them as a cache. The small CSVs the plots actually need
are tracked separately in `notebooks/paper_results/`.

## What's kept here

- **`results_extracted/results/`** — per-model reconstructions of the fixed paper patch.
  Naming: `fixed_patch_<product>_<sq|nsq>_lambda<N>_seed<N>_<rcmc|slc>_recon_phys.npy`
  (also `_linA.npy`, `_logI.png`), plus `..._sq_nsq_seed0_metrics.csv`.
  **Used by** `notebooks/make_overview_figure.py` (set `RESULTS_DIR` there if you move it).
- **`cache3_extracted/`** — full raw output of the classical-codec evaluation
  (`scripts/evaluate_classical_codec_rd.py`): per-patch CSVs, summaries, intermediate PNGs.
  The curated subset needed for the RD curves is promoted (tracked) to `notebooks/paper_results/`.

## Tracked plot inputs (in the repo)

`notebooks/paper_results/` holds the codec summary CSVs + `rd_stats` + `wandb_runs` that
`notebooks/RD-curve_plots.ipynb` and `notebooks/plot_combined_rd_curves.py` read by default —
so the RD curves regenerate from the repo alone, without this raw data.
