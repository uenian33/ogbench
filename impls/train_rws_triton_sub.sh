#!/bin/bash -l
#SBATCH --job-name=rws_ogb
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
# Use scratch for logs so $HOME stays small:
#SBATCH --output=/scratch/work/yangw4/ogbench/logs/%x_%A_%a.out
#SBATCH --error=/scratch/work/yangw4/ogbench/logs/%x_%A_%a.err
# NOTE: don't set --array here; we'll set it at submission to control parallelism.

set -euo pipefail

# === Paths / env ===
PROJECT_DIR="/scratch/work/yangw4/ogbench"
mkdir -p "$PROJECT_DIR"/{logs,data,wandb,checkpoints}

module load mamba
# Use the JAX CUDA env you already made:
source activate ogb-jax-cu12

# JAX/MuJoCo friendly env on Triton
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
# Put caches on scratch:
export OGBENCH_DATA_DIR="$PROJECT_DIR/data"
export WANDB_DIR="$PROJECT_DIR/wandb"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"

cd "$PROJECT_DIR"

# === Your tasks and discounts (index by SLURM_ARRAY_TASK_ID) ===
TASKS=(
  pointmaze-medium-navigate-v0
  pointmaze-large-navigate-v0
  pointmaze-giant-navigate-v0
  pointmaze-teleport-navigate-v0
  pointmaze-medium-stitch-v0
  pointmaze-large-stitch-v0
  pointmaze-giant-stitch-v0
  pointmaze-teleport-stitch-v0
  antmaze-medium-navigate-v0
  antmaze-large-navigate-v0
  antmaze-giant-navigate-v0
  antmaze-teleport-navigate-v0
  antmaze-medium-stitch-v0
  antmaze-large-stitch-v0
  antmaze-giant-stitch-v0
  antmaze-teleport-stitch-v0
  antmaze-medium-explore-v0
  antmaze-large-explore-v0
  antmaze-teleport-explore-v0
  humanoidmaze-medium-navigate-v0
  humanoidmaze-large-navigate-v0
  humanoidmaze-giant-navigate-v0
  humanoidmaze-medium-stitch-v0
  humanoidmaze-large-stitch-v0
  humanoidmaze-giant-stitch-v0
)

DISCOUNTS=(
  0.8
  0.8
  0.8
  0.95
  0.99
  0.999
  0.999
  0.995
  0.9
  0.9
  0.9
  0.95
  0.95
  0.99
  0.99
  0.99
  0.99
  0.995
  0.995
  0.85
  0.9
  0.85
  0.95
  0.99
  0.99
)

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
TASK="${TASKS[$IDX]}"
DISCOUNT="${DISCOUNTS[$IDX]}"

# === Build your original command (template kept intact) ===
CMD=( python main_reachability.py
  --agent_type=rws
  --dataset_type=ogbench
  --dataset_name="${TASK}"
  --hidden_dims=256,256,256
  --batch_size=128
  --discount="${DISCOUNT}"
  --num_skip_states=50
  --run_group="rws_${TASK}"
  --viz_interval=25000
  --save_interval=25000
)

echo "=================================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}   ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running task: ${TASK} (discount=${DISCOUNT})"
echo "Command: ${CMD[@]}"
echo "=================================================="

# srun is recommended under Slurm; 1 task uses 1 GPU allocated above
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-1} "${CMD[@]}"
