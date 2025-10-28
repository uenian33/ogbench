#!/bin/bash -l
#SBATCH -J rws_arr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
# If your Slurm supports VRAM constraint, uncomment next line:
# #SBATCH --gres=min-vram:10g
#SBATCH -o /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.out
#SBATCH -e /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.err

set -euo pipefail

# --- Paths / env ---
PROJECT_DIR="/scratch/work/yangw4/ogbench"
CODE_DIR="${PROJECT_DIR}/impls"
CODE_FILE="${CODE_DIR}/main_reachability.py"
ENV_NAME="ogbench"                          # matches your .conda_envs path
RUN_LIST="${1:?Usage: $0 RUN_LIST_TSV}"

# Activate env
module load mamba
source activate "${ENV_NAME}"

# JAX & caches stay under /scratch
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

# --- Minimal, robust W&B setup ---
# If WANDB_API_KEY is present in the job environment, logs go online.
# If not, we switch to offline to avoid crashes (401/no-tty).
export WANDB_ENTITY="${WANDB_ENTITY:-wenyany94}"       # or your team slug
export WANDB_PROJECT="${WANDB_PROJECT:-ogbench-rws}"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.cache/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
: "${WANDB_API_KEY:=}"          # don't fail if unset
[[ -z "$WANDB_API_KEY" ]] && export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB__SERVICE_WAIT=300

# --- Select task/discount by array index ---
INDEX="${SLURM_ARRAY_TASK_ID:?array id missing}"
mapfile -t LINES < "${RUN_LIST}"
if (( INDEX < 0 || INDEX >= ${#LINES[@]} )); then
  echo "Array index ${INDEX} out of range"
  exit 2
fi
TASK="$(echo "${LINES[$INDEX]}" | cut -f1)"
DISCOUNT="$(echo "${LINES[$INDEX]}" | cut -f2)"

# Preflight
[[ -f "${CODE_FILE}" ]] || { echo "Missing ${CODE_FILE}"; exit 2; }

cd "${CODE_DIR}"
echo "==== RWS ===="
echo "TASK=${TASK}  DISCOUNT=${DISCOUNT}"
nvidia-smi || true
echo "============="

# Command (matches your local template; just points to impls/)
python -u main_reachability.py \
  --agent_type=rws \
  --dataset_type=ogbench \
  --dataset_name="${TASK}" \
  --hidden_dims=256,256,256 \
  --batch_size=128 \
  --discount="${DISCOUNT}" \
  --num_skip_states=50 \
  --run_group="rws_${TASK}" \
  --viz_interval=25000 \
  --save_interval=25000
