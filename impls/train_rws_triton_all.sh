#!/bin/bash
# Create/submit a Slurm job array for many runs with a concurrency cap.

set -euo pipefail

# Edit if you moved things:
PROJECT_DIR="/scratch/work/yangw4/ogbench"
RUN_LIST="${PROJECT_DIR}/train_runs.tsv"
CONCURRENCY="${1:-6}"   # pass 3..6 on CLI to change; default 6

mkdir -p "${PROJECT_DIR}/logs"

# --- Write the run list (TASK \t DISCOUNT) ---
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
echo "Submitting ${N} jobs with concurrency ${CONCURRENCY}"

# Submit a single job array; Triton will auto-pick a GPU partition from --gpus
sbatch --array=0-$((N-1))%${CONCURRENCY} "${PROJECT_DIR}/train_rws_triton_sub.sh" "${RUN_LIST}"
