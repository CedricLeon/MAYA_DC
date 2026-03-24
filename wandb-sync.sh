

source /lustre/projects/1001/miniconda3/bin/activate mayadc
find logs/default -type d -name 'offline-run-*' -print0 \
  | xargs -0 -n1 wandb sync --include-offline --mark-synced
