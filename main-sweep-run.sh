#!/bin/bash
#PBS -N MAYA_DC_sweep
#PBS -q gpu4_std
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=384g:qlist=gpu

set -euo pipefail

: "${SWEEP_LMBDA:?SWEEP_LMBDA must be set via qsub -v}"
: "${SWEEP_SEED:?SWEEP_SEED must be set via qsub -v}"
: "${SWEEP_GROUP:?SWEEP_GROUP must be set via qsub -v}"

source /lustre/projects/1001/miniconda3/bin/activate mayadc
PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
LOG_DIR=$PROJECT_DIR/LOGS
mkdir -p "$LOG_DIR"

python "$PROJECT_DIR/src/train.py" \
    experiment=rcmc_compress_baseline \
    logger.wandb.group=rcmc_grid \
    model.criterion.lmbda="$SWEEP_LMBDA" \
    seed="$SWEEP_SEED" \
    hydra.run.dir="\${paths.log_dir}\${task_name}/multiruns/${SWEEP_GROUP}/lmbda_${SWEEP_LMBDA}_seed_${SWEEP_SEED}"
