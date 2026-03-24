#!/bin/bash

set -euo pipefail

PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
RUN_SCRIPT="${PROJECT_DIR}/main-classical-codec.sh"

if [[ $# -lt 3 ]]; then
    cat <<'EOF'
Usage:
  scripts/submit_classical_codec_baseline.sh <input_dir> <codec> <output_csv> [quality_csv]

Example:
  scripts/submit_classical_codec_baseline.sh \
    /lustre/projects/1001/rdelprete/SSM/data/ssm_dataset/focused/PT3 \
    jpeg \
    /lustre/projects/1001/rdelprete/MAYA_DC/notebooks/cache/jpeg_baseline_summary.csv \
    10,20,30,40,50,60,70,80,90
EOF
    exit 1
fi

INPUT_DIR="$1"
CODEC_NAME="$2"
OUTPUT_CSV="$3"
QUALITIES="${4:-10,20,30,40,50,60,70,80,90}"

RAW_CSV="${OUTPUT_CSV%.csv}_per_patch.csv"
META_JSON="${OUTPUT_CSV%.csv}_meta.json"
JOB_NAME="codec_${CODEC_NAME}"

GROUP_LABEL="$(printf '%s (2xGray, eval=slc)' "${CODEC_NAME^^}")"

job_id=$(qsub \
    -N "${JOB_NAME}" \
    -v CODEC_INPUT_DIR="${INPUT_DIR}",CODEC_NAME="${CODEC_NAME}",CODEC_OUTPUT_CSV="${OUTPUT_CSV}",QUALITIES="${QUALITIES}",REPRESENTATION=channels,EVAL_DOMAIN=slc,MAX_PRODUCTS=2,SAMPLES_PER_PROD=1000,BATCH_SIZE=4,NUM_WORKERS=0,TORCH_THREADS=1,GROUP_LABEL="${GROUP_LABEL}",RAW_CSV="${RAW_CSV}",META_JSON="${META_JSON}",OVERWRITE=1 \
    "${RUN_SCRIPT}")

printf 'Submitted %s as %s\n' "${JOB_NAME}" "${job_id}"
