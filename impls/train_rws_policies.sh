#!/bin/bash

# Flexible training script with options to select specific agents/tasks
# Usage:
#   ./run_policy_training_flexible.sh                          # Run all
#   ./run_policy_training_flexible.sh gciql                     # Run only GCIQL
#   ./run_policy_training_flexible.sh gciql gcivl               # Run GCIQL and GCIVL
#   TASKS="pointmaze-*" ./run_policy_training_flexible.sh       # Run only pointmaze tasks
#   TASK_FILTER="stitch" ./run_policy_training_flexible.sh      # Run only stitch tasks

# Configuration
BASE_DIR="exp/ReachabilityEstimation"
LOAD_EPOCH=50000
EVAL_EPISODES=1

# Agents to train (can be overridden via command line)
if [ $# -gt 0 ]; then
  AGENTS=("$@")
  echo "Training specific agents: ${AGENTS[*]}"
else
  AGENTS=("gciql" "gcivl" "hiql" "gcbc")
  echo "Training all agents: ${AGENTS[*]}"
fi

# All available tasks
ALL_TASKS=(
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

# Filter tasks based on TASK_FILTER environment variable
TASKS=()
if [ -n "$TASK_FILTER" ]; then
  echo "Filtering tasks with pattern: $TASK_FILTER"
  for task in "${ALL_TASKS[@]}"; do
    if [[ $task == *"$TASK_FILTER"* ]]; then
      TASKS+=("$task")
    fi
  done
else
  TASKS=("${ALL_TASKS[@]}")
fi

echo "Tasks to process: ${#TASKS[@]}"
echo ""

# Function to find RWS path for a given task
find_rws_path() {
  local task=$1
  local epoch=$2
  local task_dir="${BASE_DIR}/${task}_rws"
  
  if [ ! -d "$task_dir" ]; then
    echo "ERROR: Task directory not found: $task_dir" >&2
    return 1
  fi
  
  local found_paths=()
  for subdir in "$task_dir"/*/; do
    if [ -f "${subdir}params_${epoch}.pkl" ]; then
      found_paths+=("$subdir")
    fi
  done
  
  if [ ${#found_paths[@]} -eq 0 ]; then
    echo "ERROR: No valid params_${epoch}.pkl found in $task_dir" >&2
    return 1
  fi
  
  # Use most recent run
  local latest_path=$(printf '%s\n' "${found_paths[@]}" | sort -r | head -n1)
  latest_path="${latest_path%/}"
  
  echo "$latest_path"
  return 0
}

# Function to get agent-specific hyperparameters
get_agent_params() {
  local task=$1
  local agent=$2
  local params=""
  
  # Extract task type
  local task_type=""
  if [[ $task == *"pointmaze"* ]]; then
    task_type="pointmaze"
  elif [[ $task == *"antmaze"* ]]; then
    task_type="antmaze"
  elif [[ $task == *"humanoidmaze"* ]]; then
    task_type="humanoidmaze"
  fi
  
  # Check for special task categories
  local is_giant=false
  local is_stitch=false
  local is_explore=false
  
  [[ $task == *"giant"* ]] && is_giant=true
  [[ $task == *"stitch"* ]] && is_stitch=true
  [[ $task == *"explore"* ]] && is_explore=true
  
  # Agent-specific parameters
  case $agent in
    "gcbc")
      params=""
      ;;
      
    "gcivl")
      params="--agent.alpha=10.0"
      
      if [ "$is_giant" = true ]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [[ $task_type == "humanoidmaze" ]]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [ "$is_stitch" = true ]; then
        params="$params --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
      fi
      
      if [ "$is_explore" = true ]; then
        params="$params --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0"
      fi
      ;;
      
    "gciql")
      if [[ $task_type == "pointmaze" ]]; then
        params="--agent.alpha=0.003"
      elif [[ $task_type == "antmaze" ]]; then
        params="--agent.alpha=0.01"
      elif [[ $task_type == "humanoidmaze" ]]; then
        params="--agent.alpha=0.1"
      fi
      
      if [ "$is_giant" = true ]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [[ $task_type == "humanoidmaze" ]]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [ "$is_stitch" = true ]; then
        params="$params --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
      fi
      
      if [ "$is_explore" = true ]; then
        params="$params --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0"
      fi
      ;;
      
    "hiql")
      if [[ $task_type == "pointmaze" ]]; then
        params="--agent.high_alpha=3.0 --agent.low_alpha=3.0"
      elif [[ $task_type == "antmaze" ]]; then
        params="--agent.high_alpha=10.0 --agent.low_alpha=10.0"
      elif [[ $task_type == "humanoidmaze" ]]; then
        params="--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100"
      fi
      
      if [ "$is_giant" = true ]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [[ $task_type == "humanoidmaze" ]]; then
        params="$params --agent.discount=0.995"
      fi
      
      if [ "$is_stitch" = true ]; then
        params="$params --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
      fi
      
      if [ "$is_explore" = true ]; then
        params="$params --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0"
      fi
      ;;
  esac
  
  echo "$params"
}

# Main training loop
total_runs=$((${#TASKS[@]} * ${#AGENTS[@]}))
current_run=0

echo "=================================================="
echo "Starting Training Pipeline"
echo "Total planned runs: $total_runs"
echo "=================================================="
echo ""

for task in "${TASKS[@]}"; do
  echo "=================================================="
  echo "Finding RWS checkpoint for task: $task"
  echo "=================================================="
  
  RWS_PATH=$(find_rws_path "$task" "$LOAD_EPOCH")
  
  if [ $? -ne 0 ]; then
    echo "Skipping all agents for $task - no pretrained RWS params found"
    echo ""
    continue
  fi
  
  echo "✓ Found RWS path: $RWS_PATH"
  echo "✓ Loading params from: ${RWS_PATH}/params_${LOAD_EPOCH}.pkl"
  echo ""
  
  for agent in "${AGENTS[@]}"; do
    current_run=$((current_run + 1))
    
    echo "=================================================="
    echo "Run $current_run/$total_runs: Training $agent on $task"
    echo "=================================================="
    
    AGENT_PARAMS=$(get_agent_params "$task" "$agent")
    
    CMD="python main.py \
      --env_name=$task \
      --eval_episodes=$EVAL_EPISODES \
      --agent=agents/${agent}.py \
      --agent.load_rws_path=$RWS_PATH \
      --agent.load_rws_epoch=$LOAD_EPOCH \
      --agent.reachability_weighting=adv"
    
    if [ -n "$AGENT_PARAMS" ]; then
      CMD="$CMD $AGENT_PARAMS"
    fi
    
    echo "Command:"
    echo "$CMD"
    echo ""
    
    eval "$CMD"
    
    if [ $? -ne 0 ]; then
      echo "ERROR: Training failed for $agent on $task"
      read -p "Continue with next run? (y/n) " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
    fi
    
    echo ""
  done
done

echo "=================================================="
echo "All training jobs completed!"
echo "Total runs executed: $current_run"
echo "=================================================="