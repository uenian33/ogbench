#!/bin/bash

# Base command template
TEMPLATE="python main_reachability.py \
  --agent_type=rws \
  --dataset_type=ogbench \
  --dataset_name=TASK_NAME \
  --hidden_dims=256,256,256 \
  --batch_size=128 \
  --discount=DISCOUNT_VALUE \
  --num_skip_states=50 \
  --run_group=rws_TASK_NAME \
  --viz_interval=25000 \
  --save_interval=25000"

# Task names and corresponding discount values
TASKS=(
  "pointmaze-medium-navigate-v0"
  "pointmaze-large-navigate-v0"
  "pointmaze-giant-navigate-v0"
  "pointmaze-teleport-navigate-v0"
  "pointmaze-medium-stitch-v0"
  "pointmaze-large-stitch-v0"
  "pointmaze-giant-stitch-v0"
  "pointmaze-teleport-stitch-v0"
  "antmaze-medium-navigate-v0"
  "antmaze-large-navigate-v0"
  "antmaze-giant-navigate-v0"
  "antmaze-teleport-navigate-v0"
  "antmaze-medium-stitch-v0"
  "antmaze-large-stitch-v0"
  "antmaze-giant-stitch-v0"
  "antmaze-teleport-stitch-v0"
  "antmaze-medium-explore-v0"
  "antmaze-large-explore-v0"
  "antmaze-teleport-explore-v0"
  "humanoidmaze-medium-navigate-v0"
  "humanoidmaze-large-navigate-v0"
  "humanoidmaze-giant-navigate-v0"
  "humanoidmaze-medium-stitch-v0"
  "humanoidmaze-large-stitch-v0"
  "humanoidmaze-giant-stitch-v0"
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

# Run each task sequentially
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  DISCOUNT="${DISCOUNTS[$i]}"
  
  # Substitute values into template
  CMD="${TEMPLATE//TASK_NAME/$TASK}"
  CMD="${CMD//DISCOUNT_VALUE/$DISCOUNT}"
  
  echo "=================================================="
  echo "Running task $((i+1))/${#TASKS[@]}: $TASK (discount=$DISCOUNT)"
  echo "=================================================="
  echo "$CMD"
  echo ""
  
  # Execute the command
  eval "$CMD"
  
  # Check if command succeeded
  if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for $TASK"
    exit 1
  fi
  
  echo ""
done

echo "All training jobs completed successfully!"