#!/bin/bash

set -euo pipefail

PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
LOG_DIR=$PROJECT_DIR/LOGS
GRID_CFG=$PROJECT_DIR/configs/hparams_search/rcmc_grid.yaml
RUN_SCRIPT=$PROJECT_DIR/main-sweep-run.sh
SWEEP_GROUP=$(date +%Y-%m-%d_%H-%M-%S)

mkdir -p "$LOG_DIR"

lambda_csv=$(awk -F': ' '/model\.criterion\.lmbda:/ {gsub(/ /, "", $2); print $2}' "$GRID_CFG")
seed_csv=$(awk -F': ' '/seed:/ {gsub(/ /, "", $2); print $2}' "$GRID_CFG")

if [[ -z "$lambda_csv" || -z "$seed_csv" ]]; then
    echo "Failed to parse sweep grid from $GRID_CFG" >&2
    exit 1
fi

IFS=',' read -r -a lambda_values <<< "$lambda_csv"
IFS=',' read -r -a seed_values <<< "$seed_csv"

total=0
for lmbda in "${lambda_values[@]}"; do
    for seed in "${seed_values[@]}"; do
        job_name="MAYA_l${lmbda}_s${seed}"
        job_id=$(qsub \
            -N "$job_name" \
            -v SWEEP_LMBDA="$lmbda",SWEEP_SEED="$seed",SWEEP_GROUP="$SWEEP_GROUP" \
            "$RUN_SCRIPT")
        total=$((total + 1))
        printf 'Submitted %s as %s\n' "$job_name" "$job_id"
    done
done

printf 'Submitted %d sweep jobs for group %s\n' "$total" "$SWEEP_GROUP"
