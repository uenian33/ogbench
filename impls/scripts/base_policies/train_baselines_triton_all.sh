#!/bin/bash
# Submit a Slurm job array for baseline policy training across all tasks with 5 seeds each
# Usage: ./train_baselines_triton_all.sh [CONCURRENCY]

set -euo pipefail

PROJECT_DIR="/scratch/work/yangw4/ogbench"
SCRIPT_DIR="${PROJECT_DIR}/impls/scripts/base_policies"
SUB="${SCRIPT_DIR}/train_baselines_triton_sub.sh"
RUN_LIST="${SCRIPT_DIR}/train_runs.tsv"
CONCURRENCY="${1:-50}"   # default to 50 concurrent jobs

mkdir -p "${SCRIPT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Generate TSV: Task \t Agent \t Seed \t Alpha \t Discount \t ActorPRandomGoal \t ActorPTrajGoal \t ExtraArgs
cat > "${RUN_LIST}" <<'TSV'
pointmaze-medium-navigate-v0	gcbc	0	-	0.99	-	-	-
pointmaze-medium-navigate-v0	gcbc	1	-	0.99	-	-	-
pointmaze-medium-navigate-v0	gcbc	2	-	0.99	-	-	-
pointmaze-medium-navigate-v0	gcbc	3	-	0.99	-	-	-
pointmaze-medium-navigate-v0	gcbc	4	-	0.99	-	-	-
pointmaze-medium-navigate-v0	gcivl	0	10.0	0.99	-	-	-
pointmaze-medium-navigate-v0	gcivl	1	10.0	0.99	-	-	-
pointmaze-medium-navigate-v0	gcivl	2	10.0	0.99	-	-	-
pointmaze-medium-navigate-v0	gcivl	3	10.0	0.99	-	-	-
pointmaze-medium-navigate-v0	gcivl	4	10.0	0.99	-	-	-
pointmaze-medium-navigate-v0	gciql	0	0.003	0.99	-	-	-
pointmaze-medium-navigate-v0	gciql	1	0.003	0.99	-	-	-
pointmaze-medium-navigate-v0	gciql	2	0.003	0.99	-	-	-
pointmaze-medium-navigate-v0	gciql	3	0.003	0.99	-	-	-
pointmaze-medium-navigate-v0	gciql	4	0.003	0.99	-	-	-
pointmaze-medium-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-navigate-v0	gcbc	0	-	0.99	-	-	-
pointmaze-large-navigate-v0	gcbc	1	-	0.99	-	-	-
pointmaze-large-navigate-v0	gcbc	2	-	0.99	-	-	-
pointmaze-large-navigate-v0	gcbc	3	-	0.99	-	-	-
pointmaze-large-navigate-v0	gcbc	4	-	0.99	-	-	-
pointmaze-large-navigate-v0	gcivl	0	10.0	0.99	-	-	-
pointmaze-large-navigate-v0	gcivl	1	10.0	0.99	-	-	-
pointmaze-large-navigate-v0	gcivl	2	10.0	0.99	-	-	-
pointmaze-large-navigate-v0	gcivl	3	10.0	0.99	-	-	-
pointmaze-large-navigate-v0	gcivl	4	10.0	0.99	-	-	-
pointmaze-large-navigate-v0	gciql	0	0.003	0.99	-	-	-
pointmaze-large-navigate-v0	gciql	1	0.003	0.99	-	-	-
pointmaze-large-navigate-v0	gciql	2	0.003	0.99	-	-	-
pointmaze-large-navigate-v0	gciql	3	0.003	0.99	-	-	-
pointmaze-large-navigate-v0	gciql	4	0.003	0.99	-	-	-
pointmaze-large-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-navigate-v0	gcbc	0	-	0.995	-	-	-
pointmaze-giant-navigate-v0	gcbc	1	-	0.995	-	-	-
pointmaze-giant-navigate-v0	gcbc	2	-	0.995	-	-	-
pointmaze-giant-navigate-v0	gcbc	3	-	0.995	-	-	-
pointmaze-giant-navigate-v0	gcbc	4	-	0.995	-	-	-
pointmaze-giant-navigate-v0	gcivl	0	10.0	0.995	-	-	-
pointmaze-giant-navigate-v0	gcivl	1	10.0	0.995	-	-	-
pointmaze-giant-navigate-v0	gcivl	2	10.0	0.995	-	-	-
pointmaze-giant-navigate-v0	gcivl	3	10.0	0.995	-	-	-
pointmaze-giant-navigate-v0	gcivl	4	10.0	0.995	-	-	-
pointmaze-giant-navigate-v0	gciql	0	0.003	0.995	-	-	-
pointmaze-giant-navigate-v0	gciql	1	0.003	0.995	-	-	-
pointmaze-giant-navigate-v0	gciql	2	0.003	0.995	-	-	-
pointmaze-giant-navigate-v0	gciql	3	0.003	0.995	-	-	-
pointmaze-giant-navigate-v0	gciql	4	0.003	0.995	-	-	-
pointmaze-giant-navigate-v0	hiql	0	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-navigate-v0	hiql	1	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-navigate-v0	hiql	2	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-navigate-v0	hiql	3	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-navigate-v0	hiql	4	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-navigate-v0	gcbc	0	-	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcbc	1	-	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcbc	2	-	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcbc	3	-	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcbc	4	-	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcivl	0	10.0	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcivl	1	10.0	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcivl	2	10.0	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcivl	3	10.0	0.99	-	-	-
pointmaze-teleport-navigate-v0	gcivl	4	10.0	0.99	-	-	-
pointmaze-teleport-navigate-v0	gciql	0	0.003	0.99	-	-	-
pointmaze-teleport-navigate-v0	gciql	1	0.003	0.99	-	-	-
pointmaze-teleport-navigate-v0	gciql	2	0.003	0.99	-	-	-
pointmaze-teleport-navigate-v0	gciql	3	0.003	0.99	-	-	-
pointmaze-teleport-navigate-v0	gciql	4	0.003	0.99	-	-	-
pointmaze-teleport-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-stitch-v0	gcbc	0	-	0.99	-	-	-
pointmaze-medium-stitch-v0	gcbc	1	-	0.99	-	-	-
pointmaze-medium-stitch-v0	gcbc	2	-	0.99	-	-	-
pointmaze-medium-stitch-v0	gcbc	3	-	0.99	-	-	-
pointmaze-medium-stitch-v0	gcbc	4	-	0.99	-	-	-
pointmaze-medium-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gciql	0	0.003	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gciql	1	0.003	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gciql	2	0.003	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gciql	3	0.003	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	gciql	4	0.003	0.99	0.5	0.5	-
pointmaze-medium-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-medium-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-stitch-v0	gcbc	0	-	0.99	-	-	-
pointmaze-large-stitch-v0	gcbc	1	-	0.99	-	-	-
pointmaze-large-stitch-v0	gcbc	2	-	0.99	-	-	-
pointmaze-large-stitch-v0	gcbc	3	-	0.99	-	-	-
pointmaze-large-stitch-v0	gcbc	4	-	0.99	-	-	-
pointmaze-large-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gciql	0	0.003	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gciql	1	0.003	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gciql	2	0.003	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gciql	3	0.003	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	gciql	4	0.003	0.99	0.5	0.5	-
pointmaze-large-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-large-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-stitch-v0	gcbc	0	-	0.995	-	-	-
pointmaze-giant-stitch-v0	gcbc	1	-	0.995	-	-	-
pointmaze-giant-stitch-v0	gcbc	2	-	0.995	-	-	-
pointmaze-giant-stitch-v0	gcbc	3	-	0.995	-	-	-
pointmaze-giant-stitch-v0	gcbc	4	-	0.995	-	-	-
pointmaze-giant-stitch-v0	gcivl	0	10.0	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gcivl	1	10.0	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gcivl	2	10.0	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gcivl	3	10.0	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gcivl	4	10.0	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gciql	0	0.003	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gciql	1	0.003	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gciql	2	0.003	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gciql	3	0.003	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	gciql	4	0.003	0.995	0.5	0.5	-
pointmaze-giant-stitch-v0	hiql	0	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-stitch-v0	hiql	1	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-stitch-v0	hiql	2	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-stitch-v0	hiql	3	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-giant-stitch-v0	hiql	4	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-stitch-v0	gcbc	0	-	0.99	-	-	-
pointmaze-teleport-stitch-v0	gcbc	1	-	0.99	-	-	-
pointmaze-teleport-stitch-v0	gcbc	2	-	0.99	-	-	-
pointmaze-teleport-stitch-v0	gcbc	3	-	0.99	-	-	-
pointmaze-teleport-stitch-v0	gcbc	4	-	0.99	-	-	-
pointmaze-teleport-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gciql	0	0.003	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gciql	1	0.003	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gciql	2	0.003	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gciql	3	0.003	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	gciql	4	0.003	0.99	0.5	0.5	-
pointmaze-teleport-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
pointmaze-teleport-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-navigate-v0	gcbc	0	-	0.99	-	-	-
antmaze-medium-navigate-v0	gcbc	1	-	0.99	-	-	-
antmaze-medium-navigate-v0	gcbc	2	-	0.99	-	-	-
antmaze-medium-navigate-v0	gcbc	3	-	0.99	-	-	-
antmaze-medium-navigate-v0	gcbc	4	-	0.99	-	-	-
antmaze-medium-navigate-v0	gcivl	0	10.0	0.99	-	-	-
antmaze-medium-navigate-v0	gcivl	1	10.0	0.99	-	-	-
antmaze-medium-navigate-v0	gcivl	2	10.0	0.99	-	-	-
antmaze-medium-navigate-v0	gcivl	3	10.0	0.99	-	-	-
antmaze-medium-navigate-v0	gcivl	4	10.0	0.99	-	-	-
antmaze-medium-navigate-v0	gciql	0	0.3	0.99	-	-	-
antmaze-medium-navigate-v0	gciql	1	0.3	0.99	-	-	-
antmaze-medium-navigate-v0	gciql	2	0.3	0.99	-	-	-
antmaze-medium-navigate-v0	gciql	3	0.3	0.99	-	-	-
antmaze-medium-navigate-v0	gciql	4	0.3	0.99	-	-	-
antmaze-medium-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-navigate-v0	gcbc	0	-	0.99	-	-	-
antmaze-large-navigate-v0	gcbc	1	-	0.99	-	-	-
antmaze-large-navigate-v0	gcbc	2	-	0.99	-	-	-
antmaze-large-navigate-v0	gcbc	3	-	0.99	-	-	-
antmaze-large-navigate-v0	gcbc	4	-	0.99	-	-	-
antmaze-large-navigate-v0	gcivl	0	10.0	0.99	-	-	-
antmaze-large-navigate-v0	gcivl	1	10.0	0.99	-	-	-
antmaze-large-navigate-v0	gcivl	2	10.0	0.99	-	-	-
antmaze-large-navigate-v0	gcivl	3	10.0	0.99	-	-	-
antmaze-large-navigate-v0	gcivl	4	10.0	0.99	-	-	-
antmaze-large-navigate-v0	gciql	0	0.3	0.99	-	-	-
antmaze-large-navigate-v0	gciql	1	0.3	0.99	-	-	-
antmaze-large-navigate-v0	gciql	2	0.3	0.99	-	-	-
antmaze-large-navigate-v0	gciql	3	0.3	0.99	-	-	-
antmaze-large-navigate-v0	gciql	4	0.3	0.99	-	-	-
antmaze-large-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-navigate-v0	gcbc	0	-	0.995	-	-	-
antmaze-giant-navigate-v0	gcbc	1	-	0.995	-	-	-
antmaze-giant-navigate-v0	gcbc	2	-	0.995	-	-	-
antmaze-giant-navigate-v0	gcbc	3	-	0.995	-	-	-
antmaze-giant-navigate-v0	gcbc	4	-	0.995	-	-	-
antmaze-giant-navigate-v0	gcivl	0	10.0	0.995	-	-	-
antmaze-giant-navigate-v0	gcivl	1	10.0	0.995	-	-	-
antmaze-giant-navigate-v0	gcivl	2	10.0	0.995	-	-	-
antmaze-giant-navigate-v0	gcivl	3	10.0	0.995	-	-	-
antmaze-giant-navigate-v0	gcivl	4	10.0	0.995	-	-	-
antmaze-giant-navigate-v0	gciql	0	0.3	0.995	-	-	-
antmaze-giant-navigate-v0	gciql	1	0.3	0.995	-	-	-
antmaze-giant-navigate-v0	gciql	2	0.3	0.995	-	-	-
antmaze-giant-navigate-v0	gciql	3	0.3	0.995	-	-	-
antmaze-giant-navigate-v0	gciql	4	0.3	0.995	-	-	-
antmaze-giant-navigate-v0	hiql	0	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-navigate-v0	hiql	1	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-navigate-v0	hiql	2	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-navigate-v0	hiql	3	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-navigate-v0	hiql	4	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-navigate-v0	gcbc	0	-	0.99	-	-	-
antmaze-teleport-navigate-v0	gcbc	1	-	0.99	-	-	-
antmaze-teleport-navigate-v0	gcbc	2	-	0.99	-	-	-
antmaze-teleport-navigate-v0	gcbc	3	-	0.99	-	-	-
antmaze-teleport-navigate-v0	gcbc	4	-	0.99	-	-	-
antmaze-teleport-navigate-v0	gcivl	0	10.0	0.99	-	-	-
antmaze-teleport-navigate-v0	gcivl	1	10.0	0.99	-	-	-
antmaze-teleport-navigate-v0	gcivl	2	10.0	0.99	-	-	-
antmaze-teleport-navigate-v0	gcivl	3	10.0	0.99	-	-	-
antmaze-teleport-navigate-v0	gcivl	4	10.0	0.99	-	-	-
antmaze-teleport-navigate-v0	gciql	0	0.3	0.99	-	-	-
antmaze-teleport-navigate-v0	gciql	1	0.3	0.99	-	-	-
antmaze-teleport-navigate-v0	gciql	2	0.3	0.99	-	-	-
antmaze-teleport-navigate-v0	gciql	3	0.3	0.99	-	-	-
antmaze-teleport-navigate-v0	gciql	4	0.3	0.99	-	-	-
antmaze-teleport-navigate-v0	hiql	0	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-navigate-v0	hiql	1	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-navigate-v0	hiql	2	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-navigate-v0	hiql	3	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-navigate-v0	hiql	4	-	0.99	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-stitch-v0	gcbc	0	-	0.99	-	-	-
antmaze-medium-stitch-v0	gcbc	1	-	0.99	-	-	-
antmaze-medium-stitch-v0	gcbc	2	-	0.99	-	-	-
antmaze-medium-stitch-v0	gcbc	3	-	0.99	-	-	-
antmaze-medium-stitch-v0	gcbc	4	-	0.99	-	-	-
antmaze-medium-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gciql	0	0.3	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gciql	1	0.3	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gciql	2	0.3	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gciql	3	0.3	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	gciql	4	0.3	0.99	0.5	0.5	-
antmaze-medium-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-stitch-v0	gcbc	0	-	0.99	-	-	-
antmaze-large-stitch-v0	gcbc	1	-	0.99	-	-	-
antmaze-large-stitch-v0	gcbc	2	-	0.99	-	-	-
antmaze-large-stitch-v0	gcbc	3	-	0.99	-	-	-
antmaze-large-stitch-v0	gcbc	4	-	0.99	-	-	-
antmaze-large-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gciql	0	0.3	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gciql	1	0.3	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gciql	2	0.3	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gciql	3	0.3	0.99	0.5	0.5	-
antmaze-large-stitch-v0	gciql	4	0.3	0.99	0.5	0.5	-
antmaze-large-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-large-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-stitch-v0	gcbc	0	-	0.995	-	-	-
antmaze-giant-stitch-v0	gcbc	1	-	0.995	-	-	-
antmaze-giant-stitch-v0	gcbc	2	-	0.995	-	-	-
antmaze-giant-stitch-v0	gcbc	3	-	0.995	-	-	-
antmaze-giant-stitch-v0	gcbc	4	-	0.995	-	-	-
antmaze-giant-stitch-v0	gcivl	0	10.0	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gcivl	1	10.0	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gcivl	2	10.0	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gcivl	3	10.0	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gcivl	4	10.0	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gciql	0	0.3	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gciql	1	0.3	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gciql	2	0.3	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gciql	3	0.3	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	gciql	4	0.3	0.995	0.5	0.5	-
antmaze-giant-stitch-v0	hiql	0	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-stitch-v0	hiql	1	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-stitch-v0	hiql	2	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-stitch-v0	hiql	3	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-giant-stitch-v0	hiql	4	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-stitch-v0	gcbc	0	-	0.99	-	-	-
antmaze-teleport-stitch-v0	gcbc	1	-	0.99	-	-	-
antmaze-teleport-stitch-v0	gcbc	2	-	0.99	-	-	-
antmaze-teleport-stitch-v0	gcbc	3	-	0.99	-	-	-
antmaze-teleport-stitch-v0	gcbc	4	-	0.99	-	-	-
antmaze-teleport-stitch-v0	gcivl	0	10.0	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gcivl	1	10.0	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gcivl	2	10.0	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gcivl	3	10.0	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gcivl	4	10.0	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gciql	0	0.3	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gciql	1	0.3	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gciql	2	0.3	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gciql	3	0.3	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	gciql	4	0.3	0.99	0.5	0.5	-
antmaze-teleport-stitch-v0	hiql	0	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-stitch-v0	hiql	1	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-stitch-v0	hiql	2	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-stitch-v0	hiql	3	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-teleport-stitch-v0	hiql	4	-	0.99	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0
antmaze-medium-explore-v0	gcbc	0	-	0.99	-	-	-
antmaze-medium-explore-v0	gcbc	1	-	0.99	-	-	-
antmaze-medium-explore-v0	gcbc	2	-	0.99	-	-	-
antmaze-medium-explore-v0	gcbc	3	-	0.99	-	-	-
antmaze-medium-explore-v0	gcbc	4	-	0.99	-	-	-
antmaze-medium-explore-v0	gcivl	0	10.0	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gcivl	1	10.0	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gcivl	2	10.0	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gcivl	3	10.0	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gcivl	4	10.0	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gciql	0	0.01	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gciql	1	0.01	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gciql	2	0.01	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gciql	3	0.01	0.99	1.0	0.0	-
antmaze-medium-explore-v0	gciql	4	0.01	0.99	1.0	0.0	-
antmaze-medium-explore-v0	hiql	0	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-medium-explore-v0	hiql	1	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-medium-explore-v0	hiql	2	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-medium-explore-v0	hiql	3	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-medium-explore-v0	hiql	4	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-large-explore-v0	gcbc	0	-	0.99	-	-	-
antmaze-large-explore-v0	gcbc	1	-	0.99	-	-	-
antmaze-large-explore-v0	gcbc	2	-	0.99	-	-	-
antmaze-large-explore-v0	gcbc	3	-	0.99	-	-	-
antmaze-large-explore-v0	gcbc	4	-	0.99	-	-	-
antmaze-large-explore-v0	gcivl	0	10.0	0.99	1.0	0.0	-
antmaze-large-explore-v0	gcivl	1	10.0	0.99	1.0	0.0	-
antmaze-large-explore-v0	gcivl	2	10.0	0.99	1.0	0.0	-
antmaze-large-explore-v0	gcivl	3	10.0	0.99	1.0	0.0	-
antmaze-large-explore-v0	gcivl	4	10.0	0.99	1.0	0.0	-
antmaze-large-explore-v0	gciql	0	0.01	0.99	1.0	0.0	-
antmaze-large-explore-v0	gciql	1	0.01	0.99	1.0	0.0	-
antmaze-large-explore-v0	gciql	2	0.01	0.99	1.0	0.0	-
antmaze-large-explore-v0	gciql	3	0.01	0.99	1.0	0.0	-
antmaze-large-explore-v0	gciql	4	0.01	0.99	1.0	0.0	-
antmaze-large-explore-v0	hiql	0	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-large-explore-v0	hiql	1	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-large-explore-v0	hiql	2	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-large-explore-v0	hiql	3	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-large-explore-v0	hiql	4	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-teleport-explore-v0	gcbc	0	-	0.99	-	-	-
antmaze-teleport-explore-v0	gcbc	1	-	0.99	-	-	-
antmaze-teleport-explore-v0	gcbc	2	-	0.99	-	-	-
antmaze-teleport-explore-v0	gcbc	3	-	0.99	-	-	-
antmaze-teleport-explore-v0	gcbc	4	-	0.99	-	-	-
antmaze-teleport-explore-v0	gcivl	0	10.0	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gcivl	1	10.0	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gcivl	2	10.0	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gcivl	3	10.0	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gcivl	4	10.0	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gciql	0	0.01	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gciql	1	0.01	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gciql	2	0.01	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gciql	3	0.01	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	gciql	4	0.01	0.99	1.0	0.0	-
antmaze-teleport-explore-v0	hiql	0	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-teleport-explore-v0	hiql	1	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-teleport-explore-v0	hiql	2	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-teleport-explore-v0	hiql	3	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
antmaze-teleport-explore-v0	hiql	4	-	0.99	1.0	0.0	--agent.high_alpha=10.0 --agent.low_alpha=10.0
humanoidmaze-medium-navigate-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcivl	0	10.0	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcivl	1	10.0	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcivl	2	10.0	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcivl	3	10.0	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gcivl	4	10.0	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gciql	0	0.1	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gciql	1	0.1	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gciql	2	0.1	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gciql	3	0.1	0.995	-	-	-
humanoidmaze-medium-navigate-v0	gciql	4	0.1	0.995	-	-	-
humanoidmaze-medium-navigate-v0	hiql	0	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-navigate-v0	hiql	1	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-navigate-v0	hiql	2	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-navigate-v0	hiql	3	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-navigate-v0	hiql	4	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-navigate-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcivl	0	10.0	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcivl	1	10.0	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcivl	2	10.0	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcivl	3	10.0	0.995	-	-	-
humanoidmaze-large-navigate-v0	gcivl	4	10.0	0.995	-	-	-
humanoidmaze-large-navigate-v0	gciql	0	0.1	0.995	-	-	-
humanoidmaze-large-navigate-v0	gciql	1	0.1	0.995	-	-	-
humanoidmaze-large-navigate-v0	gciql	2	0.1	0.995	-	-	-
humanoidmaze-large-navigate-v0	gciql	3	0.1	0.995	-	-	-
humanoidmaze-large-navigate-v0	gciql	4	0.1	0.995	-	-	-
humanoidmaze-large-navigate-v0	hiql	0	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-navigate-v0	hiql	1	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-navigate-v0	hiql	2	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-navigate-v0	hiql	3	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-navigate-v0	hiql	4	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-navigate-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcivl	0	10.0	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcivl	1	10.0	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcivl	2	10.0	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcivl	3	10.0	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gcivl	4	10.0	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gciql	0	0.1	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gciql	1	0.1	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gciql	2	0.1	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gciql	3	0.1	0.995	-	-	-
humanoidmaze-giant-navigate-v0	gciql	4	0.1	0.995	-	-	-
humanoidmaze-giant-navigate-v0	hiql	0	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-navigate-v0	hiql	1	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-navigate-v0	hiql	2	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-navigate-v0	hiql	3	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-navigate-v0	hiql	4	-	0.995	-	-	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-stitch-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-medium-stitch-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-medium-stitch-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-medium-stitch-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-medium-stitch-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-medium-stitch-v0	gcivl	0	10.0	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gcivl	1	10.0	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gcivl	2	10.0	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gcivl	3	10.0	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gcivl	4	10.0	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gciql	0	0.1	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gciql	1	0.1	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gciql	2	0.1	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gciql	3	0.1	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	gciql	4	0.1	0.995	0.5	0.5	-
humanoidmaze-medium-stitch-v0	hiql	0	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-stitch-v0	hiql	1	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-stitch-v0	hiql	2	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-stitch-v0	hiql	3	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-medium-stitch-v0	hiql	4	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-stitch-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-large-stitch-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-large-stitch-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-large-stitch-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-large-stitch-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-large-stitch-v0	gcivl	0	10.0	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gcivl	1	10.0	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gcivl	2	10.0	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gcivl	3	10.0	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gcivl	4	10.0	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gciql	0	0.1	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gciql	1	0.1	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gciql	2	0.1	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gciql	3	0.1	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	gciql	4	0.1	0.995	0.5	0.5	-
humanoidmaze-large-stitch-v0	hiql	0	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-stitch-v0	hiql	1	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-stitch-v0	hiql	2	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-stitch-v0	hiql	3	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-large-stitch-v0	hiql	4	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-stitch-v0	gcbc	0	-	0.995	-	-	-
humanoidmaze-giant-stitch-v0	gcbc	1	-	0.995	-	-	-
humanoidmaze-giant-stitch-v0	gcbc	2	-	0.995	-	-	-
humanoidmaze-giant-stitch-v0	gcbc	3	-	0.995	-	-	-
humanoidmaze-giant-stitch-v0	gcbc	4	-	0.995	-	-	-
humanoidmaze-giant-stitch-v0	gcivl	0	10.0	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gcivl	1	10.0	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gcivl	2	10.0	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gcivl	3	10.0	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gcivl	4	10.0	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gciql	0	0.1	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gciql	1	0.1	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gciql	2	0.1	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gciql	3	0.1	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	gciql	4	0.1	0.995	0.5	0.5	-
humanoidmaze-giant-stitch-v0	hiql	0	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-stitch-v0	hiql	1	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-stitch-v0	hiql	2	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-stitch-v0	hiql	3	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
humanoidmaze-giant-stitch-v0	hiql	4	-	0.995	0.5	0.5	--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100
TSV

N=$(grep -cve '^\s*$' "${RUN_LIST}")
echo "======================================"
echo "Submitting ${N} baseline training jobs"
echo "Concurrency: ${CONCURRENCY} parallel jobs"
echo "Agents: GCBC, GCIVL, GCIQL, HIQL"
echo "Seeds: 0-4 (5 seeds per task-agent pair)"
echo "======================================"

sbatch --array=0-$((N-1))%${CONCURRENCY} "${SUB}" "${RUN_LIST}"