#!/bin/bash -l
#SBATCH -J rws_arr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=min-vram:10g
#SBATCH -o /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.out
#SBATCH -e /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/work/yangw4/ogbench"
CODE_DIR="${PROJECT_DIR}/impls"
CODE_FILE="${CODE_DIR}/main_reachability.py"
ENV_NAME="ogbench"
RUN_LIST="${1:?Usage: $0 RUN_LIST_TSV}"

# Preflight
if [[ ! -f "${CODE_FILE}" ]]; then
  echo "ERROR: ${CODE_FILE} not found."
  exit 2
fi

module load mamba
source activate "${ENV_NAME}"

# JAX/MuJoCo env
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export OGBENCH_DATA_DIR="${PROJECT_DIR}/.ogbench/data"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"   # keep caches off $HOME
mkdir -p "${OGBENCH_DATA_DIR}" "${PROJECT_DIR}/logs" "${XDG_CACHE_HOME}"


# ---------- Weights & Biases (non-interactive) ----------
set -a
# Loads WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT if present
source /scratch/work/yangw4/.secrets/wandb.env || true
set +a

# Hard defaults if not provided; prevents 404 on a wrong team slug
export WANDB_ENTITY="${WANDB_ENTITY:-wenyany94}"     # <- your user
export WANDB_PROJECT="${WANDB_PROJECT:-ogbench-rws}"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.cache/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# If no API key is loaded, run offline so training still proceeds
[[ -z "${WANDB_API_KEY:-}" ]] && export WANDB_MODE=offline

# (optional noise reduction)
export WANDB_SILENT=true
export WANDB__SERVICE_WAIT=300

# --- Select task/discount from array index ---
INDEX="${SLURM_ARRAY_TASK_ID:?array id missing}"
mapfile -t LINES < "${RUN_LIST}"
if (( INDEX < 0 || INDEX >= ${#LINES[@]} )); then
  echo "Array index ${INDEX} out of range (0..$(( ${#LINES[@]} - 1 )))"
  exit 2
fi

LINE="${LINES[$INDEX]}"
TASK="$(echo "${LINE}" | cut -f1)"
DISCOUNT="$(echo "${LINE}" | cut -f2)"

cd "${CODE_DIR}"

echo "========= RWS training (idx ${INDEX}) ========="
echo "TASK=${TASK}  DISCOUNT=${DISCOUNT}"
echo "Host: $(hostname)"
nvidia-smi || true
echo "==============================================="

# --- Command template (points to impls/main_reachability.py now) ---
TEMPLATE='python -u main_reachability.py \
  --agent_type=rws \
  --dataset_type=ogbench \
  --dataset_name=TASK_NAME \
  --hidden_dims=256,256,256 \
  --batch_size=128 \
  --discount=DISCOUNT_VALUE \
  --num_skip_states=50 \
  --run_group=rws_TASK_NAME \
  --viz_interval=25000 \
  --save_interval=25000'

CMD="${TEMPLATE//TASK_NAME/${TASK}}"
CMD="${CMD//DISCOUNT_VALUE/${DISCOUNT}}"

echo "Running: ${CMD}"
# Use srun for proper Slurm accounting + GPU binding
srun bash -lc "${CMD}"
