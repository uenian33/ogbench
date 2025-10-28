#!/bin/bash -l
#SBATCH -J rws_arr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                 # system RAM
#SBATCH --gpus=1                  # 1 GPU
#SBATCH --gres=min-vram:10g       # need ≥10 GB VRAM
#SBATCH -o /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.out
#SBATCH -e /scratch/work/yangw4/ogbench/logs/%x.%A.%a.%j.err

set -euo pipefail

# --- Paths & env ---
PROJECT_DIR="/scratch/work/yangw4/ogbench"
ENV_NAME="ogb-jax-cu12"     # change if your env has another name
RUN_LIST="${1:?Usage: $0 RUN_LIST_TSV}"

module load mamba
source activate "${ENV_NAME}"

# JAX/MuJoCo niceties
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export OGBENCH_DATA_DIR="${PROJECT_DIR}/.ogbench/data"
mkdir -p "${OGBENCH_DATA_DIR}"

# --- Pick task/discount from array index ---
INDEX="${SLURM_ARRAY_TASK_ID:?array id missing}"
mapfile -t LINES < "${RUN_LIST}"
if (( INDEX < 0 || INDEX >= ${#LINES[@]} )); then
  echo "Array index ${INDEX} out of range (0..$(( ${#LINES[@]} - 1 )))"
  exit 2
fi

LINE="${LINES[$INDEX]}"
TASK="$(echo "${LINE}" | cut -f1)"
DISCOUNT="$(echo "${LINE}" | cut -f2)"

cd "${PROJECT_DIR}"

echo "========= RWS training (idx ${INDEX}) ========="
echo "TASK=${TASK}  DISCOUNT=${DISCOUNT}"
echo "Host: $(hostname)"
nvidia-smi || true
echo "==============================================="

# --- Command template (matches your local script) ---
TEMPLATE='python main_reachability.py \
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

# Use srun for proper Slurm accounting
echo "Running: ${CMD}"
srun bash -lc "${CMD}"
