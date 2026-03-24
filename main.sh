#!/bin/bash
#PBS -N MAYA_DC
#PBS -q gpu4_std
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=384g:qlist=gpu

set -e
source /lustre/projects/1001/miniconda3/bin/activate mayadc
PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
LOG_DIR=$PROJECT_DIR/LOGS
mkdir -p $LOG_DIR

python $PROJECT_DIR/src/train.py experiment=rcmc_compress_baseline
