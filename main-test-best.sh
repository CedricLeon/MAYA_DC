#!/bin/bash
#PBS -N MAYA_DC_test
#PBS -q gpu4_std
#PBS -l walltime=01:00:00
#PBS -l select=1:ngpus=1:ncpus=16:mem=64g:qlist=gpu

set -e
source /lustre/projects/1001/miniconda3/bin/activate mayadc
PROJECT_DIR=/lustre/projects/1001/rdelprete/MAYA_DC
LOG_DIR=$PROJECT_DIR/LOGS
mkdir -p $LOG_DIR

python $PROJECT_DIR/src/train.py \
  experiment=rcmc_compress_baseline \
  train=False \
  test=True \
  ckpt_path=$PROJECT_DIR/logs/default/runs/2026-03-19_19-25-49/checkpoints/epoch_005_loss_0.0102.ckpt \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto
