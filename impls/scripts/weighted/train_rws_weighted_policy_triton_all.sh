#!/bin/bash
# Submit a Slurm job array for RWS-weighted policy training

set -euo pipefail

PROJECT_DIR="/scratch/work/yangw4/ogbench"
SUB="${PROJECT_DIR}/impls/scripts/weighted/train_rws_weighted_policy_triton_sub.sh"
RUN_LIST="${PROJECT_DIR}/rws_weighted_runs.tsv"
CONCURRENCY="${1:-6}"   # how many jobs in parallel

mkdir -p "${PROJECT_DIR}/logs"

# TSV format: Task \t Agent \t Seed \t Alpha \t Discount \t RWSSubdir \t RWSEpoch \t ReachWeighting \t ExtraArgs
# 
# RWSSubdir: Just the subdirectory name (e.g., "sd042_s_12899477.0.20251028_185233")
# RWSEpoch: Must match params_*.pkl file (typically 400000)
# ReachWeighting options: vanilla, exponential, indicator, linear

# Example TSV - REPLACE SUBDIRECTORY NAMES WITH YOUR ACTUAL CHECKPOINT DIRECTORIES
cat > "${RUN_LIST}" <<'TSV'
# Task	Agent	Seed	Alpha	Discount	RWSSubdir	RWSEpoch	ReachWeighting	ExtraArgs
antmaze-giant-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899503.0.20251028_185615	400000	vanilla	-
antmaze-giant-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899503.0.20251028_185615	400000	vanilla	-
antmaze-giant-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899503.0.20251028_185615	400000	vanilla	-
antmaze-giant-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899749.0.20251028_191603	400000	vanilla	-
antmaze-giant-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899749.0.20251028_191603	400000	vanilla	-
antmaze-giant-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899749.0.20251028_191603	400000	vanilla	-
antmaze-large-explore-v0	gciql	0	0.003	0.995	sd042_s_12900048.0.20251028_193305	400000	vanilla	-
antmaze-large-explore-v0	gciql	0	0.003	0.995	sd042_s_12900048.0.20251028_193305	400000	vanilla	-
antmaze-large-explore-v0	gciql	0	0.003	0.995	sd042_s_12900048.0.20251028_193305	400000	vanilla	-
antmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899480.0.20251028_185404	400000	vanilla	-
antmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899480.0.20251028_185404	400000	vanilla	-
antmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899480.0.20251028_185404	400000	vanilla	-
antmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899665.0.20251028_190125	400000	vanilla	-
antmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899665.0.20251028_190125	400000	vanilla	-
antmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899665.0.20251028_190125	400000	vanilla	-
antmaze-medium-explore-v0	gciql	0	0.003	0.99	sd042_s_12900041.0.20251028_193120	400000	vanilla	-
antmaze-medium-explore-v0	gciql	0	0.003	0.99	sd042_s_12900041.0.20251028_193120	400000	vanilla	-
antmaze-medium-explore-v0	gciql	0	0.003	0.99	sd042_s_12900041.0.20251028_193120	400000	vanilla	-
antmaze-medium-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899477.0.20251028_185233	400000	vanilla	-
antmaze-medium-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899477.0.20251028_185233	400000	vanilla	-
antmaze-medium-navigate-v0	gciql	0	0.003	0.9	sd042_s_12899477.0.20251028_185233	400000	vanilla	-
antmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12899656.0.20251028_190020	400000	vanilla	-
antmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12899656.0.20251028_190020	400000	vanilla	-
antmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12899656.0.20251028_190020	400000	vanilla	-
antmaze-teleport-explore-v0	gciql	0	0.003	0.995	sd042_s_12900050.0.20251028_193348	400000	vanilla	-
antmaze-teleport-explore-v0	gciql	0	0.003	0.995	sd042_s_12900050.0.20251028_193348	400000	vanilla	-
antmaze-teleport-explore-v0	gciql	0	0.003	0.995	sd042_s_12900050.0.20251028_193348	400000	vanilla	-
antmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12899556.0.20251028_185628	400000	vanilla	-
antmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12899556.0.20251028_185628	400000	vanilla	-
antmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12899556.0.20251028_185628	400000	vanilla	-
antmaze-teleport-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899961.0.20251028_192953	400000	vanilla	-
antmaze-teleport-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899961.0.20251028_192953	400000	vanilla	-
antmaze-teleport-stitch-v0	gciql	0	0.003	0.99	sd042_s_12899961.0.20251028_192953	400000	vanilla	-
humanoidmaze-giant-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900163.0.20251028_195115	400000	vanilla	-
humanoidmaze-giant-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900163.0.20251028_195115	400000	vanilla	-
humanoidmaze-giant-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900163.0.20251028_195115	400000	vanilla	-
humanoidmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12900070.0.20251028_193712	400000	vanilla	-
humanoidmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12900070.0.20251028_193712	400000	vanilla	-
humanoidmaze-large-navigate-v0	gciql	0	0.003	0.9	sd042_s_12900070.0.20251028_193712	400000	vanilla	-
humanoidmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12900260.0.20251028_201003	400000	vanilla	-
humanoidmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12900260.0.20251028_201003	400000	vanilla	-
humanoidmaze-large-stitch-v0	gciql	0	0.003	0.99	sd042_s_12900260.0.20251028_201003	400000	vanilla	-
humanoidmaze-medium-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900064.0.20251028_193628	400000	vanilla	-
humanoidmaze-medium-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900064.0.20251028_193628	400000	vanilla	-
humanoidmaze-medium-navigate-v0	gciql	0	0.003	0.85	sd042_s_12900064.0.20251028_193628	400000	vanilla	-
humanoidmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12900220.0.20251028_200549	400000	vanilla	-
humanoidmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12900220.0.20251028_200549	400000	vanilla	-
humanoidmaze-medium-stitch-v0	gciql	0	0.003	0.95	sd042_s_12900220.0.20251028_200549	400000	vanilla	-
pointmaze-giant-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898330.0.20251028_182528	400000	vanilla	-
pointmaze-giant-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898330.0.20251028_182528	400000	vanilla	-
pointmaze-giant-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898330.0.20251028_182528	400000	vanilla	-
pointmaze-giant-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898334.0.20251028_182534	400000	vanilla	-
pointmaze-giant-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898334.0.20251028_182534	400000	vanilla	-
pointmaze-giant-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898334.0.20251028_182534	400000	vanilla	-
pointmaze-large-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898329.0.20251028_182547	400000	vanilla	-
pointmaze-large-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898329.0.20251028_182547	400000	vanilla	-
pointmaze-large-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898329.0.20251028_182547	400000	vanilla	-
pointmaze-large-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898333.0.20251028_182531	400000	vanilla	-
pointmaze-large-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898333.0.20251028_182531	400000	vanilla	-
pointmaze-large-stitch-v0	gciql	0	0.003	0.999	sd042_s_12898333.0.20251028_182531	400000	vanilla	-
pointmaze-medium-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898328.0.20251028_182547	400000	vanilla	-
pointmaze-medium-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898328.0.20251028_182547	400000	vanilla	-
pointmaze-medium-navigate-v0	gciql	0	0.003	0.8	sd042_s_12898328.0.20251028_182547	400000	vanilla	-
pointmaze-medium-stitch-v0	gciql	0	0.003	0.99	sd042_s_12898332.0.20251028_182530	400000	vanilla	-
pointmaze-medium-stitch-v0	gciql	0	0.003	0.99	sd042_s_12898332.0.20251028_182530	400000	vanilla	-
pointmaze-medium-stitch-v0	gciql	0	0.003	0.99	sd042_s_12898332.0.20251028_182530	400000	vanilla	-
pointmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12898331.0.20251028_182547	400000	vanilla	-
pointmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12898331.0.20251028_182547	400000	vanilla	-
pointmaze-teleport-navigate-v0	gciql	0	0.003	0.95	sd042_s_12898331.0.20251028_182547	400000	vanilla	-
pointmaze-teleport-stitch-v0	gciql	0	0.003	0.995	sd042_s_12898979.0.20251028_184524	400000	vanilla	-
pointmaze-teleport-stitch-v0	gciql	0	0.003	0.995	sd042_s_12898979.0.20251028_184524	400000	vanilla	-
pointmaze-teleport-stitch-v0	gciql	0	0.003	0.995	sd042_s_12898979.0.20251028_184524	400000	vanilla	-
TSV

# NOTE: Use find_rws_checkpoints.sh to find your actual checkpoint subdirectories!
# ./find_rws_checkpoints.sh /scratch/work/yangw4/ogbench
# 
# Or use generate_rws_weighted_config.py to automatically create the TSV:
# python generate_rws_weighted_config.py --epoch 400000 --output "${RUN_LIST}"

# Count non-comment lines
N=$(grep -cve '^\s*$' -e '^#' "${RUN_LIST}")
echo "Submitting ${N} array jobs for RWS-weighted policy training, concurrency ${CONCURRENCY}"

if [[ ${N} -eq 0 ]]; then
    echo "ERROR: No valid entries in ${RUN_LIST}"
    echo "Please update the TSV file with your actual RWS checkpoint directories"
    exit 1
fi

sbatch --array=0-$((N-1))%${CONCURRENCY} "${SUB}" "${RUN_LIST}"