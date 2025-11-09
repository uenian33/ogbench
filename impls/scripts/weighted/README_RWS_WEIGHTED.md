# RWS-Weighted Policy Training Setup

This setup allows training baseline agents (GCIQL, etc.) with pre-trained RWS networks for improved goal-conditioned reinforcement learning.

## Overview

RWS (Reachability-Weighted Sampling) provides learned reachability estimates that can guide policy training. By loading pre-trained RWS networks, baseline agents can leverage these estimates for better exploration and learning.

## Files

- **train_rws_weighted_policy_triton_sub.sh** - Slurm submission script for individual jobs
- **train_rws_weighted_policy_triton_all.sh** - Array job launcher
- **find_rws_checkpoints.sh** - Locate available RWS checkpoints
- **generate_rws_weighted_config.py** - Auto-generate TSV configurations

## Quick Start

### 1. Find Available RWS Checkpoints

```bash
./find_rws_checkpoints.sh /scratch/work/yangw4/ogbench
```

This will create `rws_checkpoints.txt` listing all available RWS models.

### 2. Generate Training Configuration

```bash
# Generate default configuration (all tasks, vanilla/exponential/indicator weightings)
python generate_rws_weighted_config.py

# Custom configuration
python generate_rws_weighted_config.py \
  --agents gciql \
  --seeds 0 1 2 \
  --weightings vanilla exponential \
  --epoch 400000 \
  --tasks antmaze-medium-navigate-v0 antmaze-large-navigate-v0
```

### 3. Submit Jobs

```bash
# Submit with default concurrency (6 parallel jobs)
sbatch --array=0-23%6 train_rws_weighted_policy_triton_sub.sh rws_weighted_runs.tsv

# Or use the all script
bash train_rws_weighted_policy_triton_all.sh 6
```

## Configuration Format

The TSV file format:
```
Task    Agent    Seed    Alpha    Discount    RWSDir    RWSEpoch    ReachWeighting    ExtraArgs
```

Example line:
```
antmaze-medium-navigate-v0    gciql    0    0.003    0.9    rws_antmaze-medium-navigate-v0_rws/sd042    400000    vanilla    -
```

## Reachability Weighting Options

- **vanilla**: Direct reachability values as weights
- **exponential**: Exponential transformation of reachability
- **indicator**: Binary (0/1) based on reachability threshold
- **linear**: Linear scaling of reachability values

## GPU Requirements

The scripts request GPUs with:
- Minimum 12GB VRAM (for loading both RWS and policy networks)
- CUDA Compute Capability 7.0+ (V100 and newer)

Adjust in submission script if needed:
```bash
#SBATCH --gres=min-vram:12g,min-cuda-cc:70
```

## Directory Structure

```
/scratch/work/yangw4/ogbench/
├── weights/
│   └── ReachabilityEstimation/
│       ├── rws_antmaze-medium-navigate-v0_rws/
│       │   └── sd042_s_12899477.0.20251028_185233/
│       │       └── epoch_400000.pkl
│       └── ...
├── exp/
│   └── OGBench-RWS/
│       └── {task}_{agent}_{weighting}/
│           └── {seed}/
└── logs/
    └── *.out, *.err
```

## Monitoring Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f /scratch/work/yangw4/ogbench/logs/*.out

# Monitor GPU usage
watch -n 2 nvidia-smi
```

## W&B Integration

Results are logged to W&B project: `ogbench-rws-weighted`

View at: https://wandb.ai/wenyany94/ogbench-rws-weighted

## Example Commands

### Train specific task with RWS
```bash
python main.py \
  --env_name=antmaze-medium-navigate-v0 \
  --agent=agents/gciql.py \
  --agent.alpha=0.003 \
  --agent.discount=0.9 \
  --agent.load_rws_path=/scratch/work/yangw4/ogbench/weights/ReachabilityEstimation/rws_antmaze-medium-navigate-v0_rws/sd042 \
  --agent.load_rws_epoch=400000 \
  --agent.reachability_weighting=vanilla
```

### Batch submission for multiple seeds
```bash
for seed in 0 1 2; do
  sbatch train_rws_weighted_policy_triton_sub.sh <<EOF
antmaze-medium-navigate-v0	gciql	${seed}	0.003	0.9	rws_antmaze-medium-navigate-v0_rws/sd042	400000	vanilla	-
EOF
done
```

## Troubleshooting

### RWS checkpoint not found
- Verify path in weights/ReachabilityEstimation/
- Check epoch number exists
- Use find_rws_checkpoints.sh to list available models

### GPU memory issues
- Increase VRAM requirement: `--gres=min-vram:16g`
- Reduce batch size in agent config

### Queue times
- Relax GPU constraints: `--gres=min-vram:10g,min-cuda-cc:70`
- Reduce concurrency in array job

## Notes

- RWS paths can be relative (from weights/ReachabilityEstimation/) or absolute
- The `sd042` in paths refers to seed directories from RWS training
- Typical RWS epochs: 100000, 200000, 300000, 400000
- Default alpha=0.003 works well for GCIQL with RWS
