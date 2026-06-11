# Running on the cluster (ESA SpaceHPC)

The paper runs were done on **ESA SpaceHPC**, which uses **PBS / OpenPBS** (not SLURM):
submit with `qsub`, monitor with `qstat`. Queues: `gpu4_std` (GPU), `cpu_std` (CPU).
Project root was `/lustre/projects/1001/rdelprete/MAYA_DC`; conda env `mayadc`.

The PBS launch scripts (`main-*.sh`, `wandb-sync.sh`,
`scripts/submit_classical_codec_baseline.sh`) were deleted in the cleanup — they hardcoded
those `/lustre` paths and are useless off that cluster. Their commands are preserved here.

PBS header used for training jobs (4×GPU node):

```bash
#PBS -q gpu4_std
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=384g:qlist=gpu
```

## Commands

```bash
# Single training run
python src/train.py experiment=rcmc_compress_baseline

# Fast debug (1 GPU)
python src/train.py experiment=rcmc_compress_baseline debug=fdr

# Test a checkpoint
python src/train.py experiment=rcmc_compress_baseline \
  train=False test=True ckpt_path=<path/to.ckpt> trainer.devices=1
```

### Paper sweep (λ × seed)

The grid lives in `configs/hparams_search/rcmc_grid.yaml` (`model.criterion.lmbda` ∈ {1…1000},
`seed` list). On the cluster each (λ, seed) was submitted as its own PBS job; the equivalent
direct command per point is:

```bash
python src/train.py experiment=rcmc_compress_baseline \
  logger.wandb.group=rcmc_grid \
  model.criterion.lmbda=<LMBDA> seed=<SEED> \
  hydra.run.dir=<logs>/multiruns/<group>/lmbda_<LMBDA>_seed_<SEED>
```

### Classical-codec baseline (JPEG / JPEG2000 / WebP)

Run on CPU (`cpu_std`). One job per codec:

```bash
python scripts/evaluate_classical_codec_rd.py <input_dir> <jpeg|jpeg2000|webp> <out.csv> \
  --representation channels --eval-domain slc \
  --patch-size 512 512 --azimuth-buffer 512 \
  --max-products 2 --samples-per-prod 1000 --batch-size 4 \
  --qualities 10,20,30,40,50,60,70,80,90 \
  --raw-csv <out>_per_patch.csv --meta-json <out>_meta.json
```

### WandB offline sync

Jobs logged offline; afterwards sync from a node with network:

```bash
find logs/default -type d -name 'offline-run-*' -print0 \
  | xargs -0 -n1 wandb sync --include-offline --mark-synced
```
