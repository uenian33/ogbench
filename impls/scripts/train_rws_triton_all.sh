#!/bin/bash
# Submit a Slurm job array; cap concurrent runs via %CONCURRENCY.

set -euo pipefail

PROJECT_DIR="/scratch/work/yangw4/ogbench"
SUB="${PROJECT_DIR}/impls/train_rws_triton_sub.sh"
RUN_LIST="${PROJECT_DIR}/train_runs.tsv"
CONCURRENCY="${1:-6}"   # how many jobs in parallel (e.g., 3..6)

mkdir -p "${PROJECT_DIR}/logs"

# Task \t Discount
cat > "${RUN_LIST}" <<'TSV'
pointmaze-medium-navigate-v0	0.8
pointmaze-large-navigate-v0	0.8
pointmaze-giant-navigate-v0	0.8
pointmaze-teleport-navigate-v0	0.95
pointmaze-medium-stitch-v0	0.99
pointmaze-large-stitch-v0	0.999
pointmaze-giant-stitch-v0	0.999
pointmaze-teleport-stitch-v0	0.995
antmaze-medium-navigate-v0	0.9
antmaze-large-navigate-v0	0.9
antmaze-giant-navigate-v0	0.9
antmaze-teleport-navigate-v0	0.95
antmaze-medium-stitch-v0	0.95
antmaze-large-stitch-v0	0.99
antmaze-giant-stitch-v0	0.99
antmaze-teleport-stitch-v0	0.99
antmaze-medium-explore-v0	0.99
antmaze-large-explore-v0	0.995
antmaze-teleport-explore-v0	0.995
humanoidmaze-medium-navigate-v0	0.85
humanoidmaze-large-navigate-v0	0.9
humanoidmaze-giant-navigate-v0	0.85
humanoidmaze-medium-stitch-v0	0.95
humanoidmaze-large-stitch-v0	0.99
humanoidmaze-giant-stitch-v0	0.99
TSV

N=$(grep -cve '^\s*$' "${RUN_LIST}")
echo "Submitting ${N} array jobs, concurrency ${CONCURRENCY}"

sbatch --array=0-$((N-1))%${CONCURRENCY} "${SUB}" "${RUN_LIST}"
