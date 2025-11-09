#!/bin/bash -l
#SBATCH -J baseline_arr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
# If your Slurm supports VRAM constraint, uncomment next line:
# #SBATCH --gres=gpu:1,vram:10g
#SBATCH -o /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.out
#SBATCH -e /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.err

set -euo pipefail

# --- Paths / env ---
PROJECT_DIR="/scratch/work/yangw4/ogbench"
CODE_DIR="${PROJECT_DIR}/impls"
CODE_FILE="${CODE_DIR}/main.py"
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
export WANDB_PROJECT=ogbench-baselines

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
mapfile -t LINES < "${RUN_LIST}"
if (( INDEX < 0 || INDEX >= ${#LINES[@]} )); then
  echo "Array index ${INDEX} out of range"
  exit 2
fi

# Parse TSV line: Task \t Agent \t Seed \t Alpha \t Discount \t ActorPRandomGoal \t ActorPTrajGoal \t ExtraArgs
LINE="${LINES[$INDEX]}"
TASK=$(echo "$LINE" | cut -f1)
AGENT=$(echo "$LINE" | cut -f2)
SEED=$(echo "$LINE" | cut -f3)
ALPHA=$(echo "$LINE" | cut -f4)
DISCOUNT=$(echo "$LINE" | cut -f5)
ACTOR_P_RANDOM=$(echo "$LINE" | cut -f6)
ACTOR_P_TRAJ=$(echo "$LINE" | cut -f7)
EXTRA_ARGS=$(echo "$LINE" | cut -f8)

# Preflight check
[[ -f "${CODE_FILE}" ]] || { echo "Missing ${CODE_FILE}"; exit 2; }

cd "${CODE_DIR}"
echo "=========================================="
echo "BASELINE TRAINING"
echo "Task:     ${TASK}"
echo "Agent:    ${AGENT}"
echo "Seed:     ${SEED}"
echo "Alpha:    ${ALPHA}"
echo "Discount: ${DISCOUNT}"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# Build command dynamically
SAVE_DIR="${PROJECT_DIR}/exp/OGBench/${TASK}_${AGENT}/${SEED}"
RUN_GROUP="${TASK}_${AGENT}_${SEED}"

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

# Add actor probabilities if not '-'
if [[ "${ACTOR_P_RANDOM}" != "-" ]]; then
  CMD="${CMD} --agent.actor_p_randomgoal=${ACTOR_P_RANDOM}"
fi

if [[ "${ACTOR_P_TRAJ}" != "-" ]]; then
  CMD="${CMD} --agent.actor_p_trajgoal=${ACTOR_P_TRAJ}"
fi

# Add extra args if not '-'
if [[ "${EXTRA_ARGS}" != "-" ]]; then
  CMD="${CMD} ${EXTRA_ARGS}"
fi

# Execute training
echo "Executing: ${CMD}"
echo "=========================================="
eval "${CMD}"