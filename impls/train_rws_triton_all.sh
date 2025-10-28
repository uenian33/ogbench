#!/bin/bash
set -euo pipefail

# How many in parallel? (3–6 per your request)
MAX_PARALLEL="${1:-6}"

# We have 25 tasks: indices 0..24
ARRAY_RANGE="0-24%${MAX_PARALLEL}"

mkdir -p /scratch/work/yangw4/ogbench/impls/logs

# Submit without pinning a partition; Triton will auto-pick a GPU partition
sbatch --array="${ARRAY_RANGE}" train_rws_triton_sub.sbatch

echo "Submitted array ${ARRAY_RANGE}. View queue with:  squeue -u $USER"


# cd /scratch/work/yangw4/ogbench
# chmod +x submit_all.sh
# ./submit_all.sh 6     # run up to 6 trainings at the same time (change to 3..6 as you like)
