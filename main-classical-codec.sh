#!/bin/bash
#PBS -N MAYA_codec
#PBS -q cpu_std
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=64g:qlist=cpu

set -euo pipefail

: "${CODEC_INPUT_DIR:?CODEC_INPUT_DIR must be set via qsub -v}"
: "${CODEC_NAME:?CODEC_NAME must be set via qsub -v}"
: "${CODEC_OUTPUT_CSV:?CODEC_OUTPUT_CSV must be set via qsub -v}"

PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
PYTHON_BIN=/lustre/home/u10010007/.conda/envs/mayadc/bin/python

QUALITIES="${QUALITIES:-}"
REPRESENTATION="${REPRESENTATION:-channels}"
EVAL_DOMAIN="${EVAL_DOMAIN:-slc}"
PATCH_AZ="${PATCH_AZ:-512}"
PATCH_RG="${PATCH_RG:-512}"
AZIMUTH_BUFFER="${AZIMUTH_BUFFER:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_PRODUCTS="${MAX_PRODUCTS:-2}"
SAMPLES_PER_PROD="${SAMPLES_PER_PROD:-1000}"
LIMIT_BATCHES="${LIMIT_BATCHES:-}"
DATASET_SHORT="${DATASET_SHORT:-MAYA4}"
GROUP_LABEL="${GROUP_LABEL:-}"
RAW_CSV="${RAW_CSV:-}"
META_JSON="${META_JSON:-}"
OVERWRITE="${OVERWRITE:-1}"
TORCH_THREADS="${TORCH_THREADS:-1}"

export OPENBLAS_NUM_THREADS="${TORCH_THREADS}"
export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"
export NUMEXPR_NUM_THREADS="${TORCH_THREADS}"
export VECLIB_MAXIMUM_THREADS="${TORCH_THREADS}"

mkdir -p "$(dirname "${CODEC_OUTPUT_CSV}")"

cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_DIR}/scripts/evaluate_classical_codec_rd.py"
    "${CODEC_INPUT_DIR}"
    "${CODEC_NAME}"
    "${CODEC_OUTPUT_CSV}"
    --representation "${REPRESENTATION}"
    --eval-domain "${EVAL_DOMAIN}"
    --patch-size "${PATCH_AZ}" "${PATCH_RG}"
    --azimuth-buffer "${AZIMUTH_BUFFER}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-products "${MAX_PRODUCTS}"
    --samples-per-prod "${SAMPLES_PER_PROD}"
    --dataset-short "${DATASET_SHORT}"
    --torch-threads "${TORCH_THREADS}"
)

if [[ -n "${QUALITIES}" ]]; then
    cmd+=(--qualities "${QUALITIES}")
fi

if [[ -n "${LIMIT_BATCHES}" ]]; then
    cmd+=(--limit-batches "${LIMIT_BATCHES}")
fi

if [[ -n "${GROUP_LABEL}" ]]; then
    cmd+=(--group-label "${GROUP_LABEL}")
fi

if [[ -n "${RAW_CSV}" ]]; then
    cmd+=(--raw-csv "${RAW_CSV}")
fi

if [[ -n "${META_JSON}" ]]; then
    cmd+=(--meta-json "${META_JSON}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
fi

printf 'Running classical codec baseline job:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
