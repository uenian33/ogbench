#!/bin/bash -l
#SBATCH -J rws_policy_arr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
# Request GPU with at least 12GB VRAM (for loading both RWS and policy networks)
#SBATCH --gres=min-vram:12g,min-cuda-cc:70
#SBATCH -o /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.out
#SBATCH -e /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.err

set -euo pipefail

# --- Paths / env ---
PROJECT_DIR="/scratch/work/yangw4/ogbench"
CODE_DIR="${PROJECT_DIR}/impls"
CODE_FILE="${CODE_DIR}/main.py"
# Base path for RWS weights - adjust if your structure is different
RWS_BASE_DIR="${PROJECT_DIR}/weights/ReachabilityEstimation"
ENV_NAME="ogbench"
RUN_LIST="${1:?Usage: $0 RUN_LIST_TSV}"

# Activate environment
module load mamba
source activate "${ENV_NAME}"

# JAX & caches stay under /scratch
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

# --- W&B setup for non-interactive environment ---
export WANDB_API_KEY=149086ee6abaf8e282c5de9163b7d3969d4c5c01
export WANDB_ENTITY=wenyany94
export WANDB_PROJECT=ogbench-rws-weighted

export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.cache/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
export WANDB_SILENT=true
export WANDB__SERVICE_WAIT=300

# Redirect ALL wandb directories to scratch (not home)
export WANDB_CONFIG_DIR="${PROJECT_DIR}/.config/wandb"
mkdir -p "$WANDB_CONFIG_DIR"

# Explicitly login to wandb in non-interactive environment
wandb login --relogin "${WANDB_API_KEY}"

# --- Parse task configuration by array index ---
INDEX="${SLURM_ARRAY_TASK_ID:?array id missing}"

# Read non-comment lines from TSV
mapfile -t LINES < <(grep -v '^#' "${RUN_LIST}" | grep -v '^\s*$')

if (( INDEX < 0 || INDEX >= ${#LINES[@]} )); then
  echo "Array index ${INDEX} out of range (${#LINES[@]} lines in TSV)"
  exit 2
fi

# Parse TSV line: Task \t Agent \t Seed \t Alpha \t Discount \t RWSSubdir \t RWSEpoch \t ReachWeighting \t ExtraArgs
LINE="${LINES[$INDEX]}"
TASK=$(echo "$LINE" | cut -f1)
AGENT=$(echo "$LINE" | cut -f2)
SEED=$(echo "$LINE" | cut -f3)
ALPHA=$(echo "$LINE" | cut -f4)
DISCOUNT=$(echo "$LINE" | cut -f5)
RWS_SUBDIR=$(echo "$LINE" | cut -f6)
RWS_EPOCH=$(echo "$LINE" | cut -f7)
REACH_WEIGHTING=$(echo "$LINE" | cut -f8)
EXTRA_ARGS=$(echo "$LINE" | cut -f9)

# Construct full RWS path
# Expected structure: .../weights/ReachabilityEstimation/rws_{task}_rws/{subdir}/params_{epoch}.pkl
FULL_RWS_PATH="${RWS_BASE_DIR}/rws_${TASK}_rws/${RWS_SUBDIR}"

# Verify RWS checkpoint exists
CHECKPOINT_FILE="${FULL_RWS_PATH}/params_${RWS_EPOCH}.pkl"
if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
    echo "WARNING: RWS checkpoint not found: ${CHECKPOINT_FILE}"
    echo "Directory contents:"
    ls -la "${FULL_RWS_PATH}/" 2>/dev/null || echo "Directory not found: ${FULL_RWS_PATH}"
    echo "Continuing anyway (training will fail if checkpoint is required)..."
fi

# Preflight check
[[ -f "${CODE_FILE}" ]] || { echo "Missing ${CODE_FILE}"; exit 2; }

cd "${CODE_DIR}"
echo "=========================================="
echo "RWS-WEIGHTED POLICY TRAINING"
echo "Task:           ${TASK}"
echo "Agent:          ${AGENT}"
echo "Seed:           ${SEED}"
echo "Alpha:          ${ALPHA}"
echo "Discount:       ${DISCOUNT}"
echo "RWS Path:       ${FULL_RWS_PATH}"
echo "RWS Checkpoint: params_${RWS_EPOCH}.pkl"
echo "Reachability:   ${REACH_WEIGHTING}"
echo "Array Index:    ${INDEX}"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# Build command dynamically
SAVE_DIR="${PROJECT_DIR}/exp/OGBench-RWS/${TASK}_${AGENT}_${REACH_WEIGHTING}/${SEED}"
RUN_GROUP="${TASK}_${AGENT}_${REACH_WEIGHTING}_${SEED}"

# Use relative path from impls directory for the load_rws_path
# Since we're running from CODE_DIR (/scratch/work/yangw4/ogbench/impls)
# and weights are in /scratch/work/yangw4/ogbench/weights/
RELATIVE_RWS_PATH="../weights/ReachabilityEstimation/rws_${TASK}_rws/${RWS_SUBDIR}"

CMD="python -u main.py \
  --env_name=${TASK} \
  --eval_episodes=50 \
  --agent=agents/${AGENT}.py \
  --save_dir=${SAVE_DIR} \
  --run_group=${RUN_GROUP} \
  --seed=${SEED}"

# Add alpha if not '-'
if [[ "${ALPHA}" != "-" ]]; then
  CMD="${CMD} --agent.alpha=${ALPHA}"
fi

# Add discount (always present in TSV)
CMD="${CMD} --agent.discount=${DISCOUNT}"

# Add RWS loading parameters using relative path
CMD="${CMD} --agent.load_rws_path=${RELATIVE_RWS_PATH}"
CMD="${CMD} --agent.load_rws_epoch=${RWS_EPOCH}"
CMD="${CMD} --agent.reachability_weighting=${REACH_WEIGHTING}"

# Add extra args if not '-'
if [[ "${EXTRA_ARGS}" != "-" ]]; then
  CMD="${CMD} ${EXTRA_ARGS}"
fi

# Execute training
echo "Executing: ${CMD}"
echo "=========================================="
eval "${CMD}"