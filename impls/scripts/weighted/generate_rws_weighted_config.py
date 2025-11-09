#!/usr/bin/env python3
"""
generate_rws_weighted_config.py - Automatically generate TSV configuration for RWS-weighted training
Generates 5 seeds (0-4) for each agent/task/weighting combination
"""

import os
import glob
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Agent-specific default parameters matching baseline configs
AGENT_CONFIGS = {
    'gcbc': {
        'alpha': None,  # No alpha for GCBC
        'extra_args': None
    },
    'gcivl': {
        'alpha': 10.0,
        'extra_args': None
    },
    'gciql': {
        'alpha': {
            'pointmaze': 0.003,
            'antmaze': 0.3,  # Different for antmaze
            'humanoidmaze': 0.1,  # Different for humanoidmaze
            'explore': 0.01,  # Special for explore tasks
        },
        'extra_args': None
    },
    'hiql': {
        'alpha': None,  # HIQL uses high_alpha and low_alpha in extra_args
        'extra_args': {
            'default': '--agent.high_alpha=3.0 --agent.low_alpha=3.0',
            'explore': '--agent.high_alpha=10.0 --agent.low_alpha=10.0',
            'humanoidmaze': '--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100'
        }
    }
}

# Task-specific actor probability settings (for stitch/explore tasks)
ACTOR_PROBS = {
    'stitch': {'actor_p_randomgoal': 0.5, 'actor_p_trajgoal': 0.5},
    'explore': {'actor_p_randomgoal': 1.0, 'actor_p_trajgoal': 0.0},
}

# Discount factors per task (from baseline configs)
DISCOUNT_FACTORS = {
    # Pointmaze
    'pointmaze-medium-navigate-v0': 0.99,
    'pointmaze-large-navigate-v0': 0.99,
    'pointmaze-giant-navigate-v0': 0.995,
    'pointmaze-teleport-navigate-v0': 0.99,
    'pointmaze-medium-stitch-v0': 0.99,
    'pointmaze-large-stitch-v0': 0.99,
    'pointmaze-giant-stitch-v0': 0.995,
    'pointmaze-teleport-stitch-v0': 0.99,
    
    # Antmaze
    'antmaze-medium-navigate-v0': 0.99,
    'antmaze-large-navigate-v0': 0.99,
    'antmaze-giant-navigate-v0': 0.995,
    'antmaze-teleport-navigate-v0': 0.99,
    'antmaze-medium-stitch-v0': 0.99,
    'antmaze-large-stitch-v0': 0.99,
    'antmaze-giant-stitch-v0': 0.995,
    'antmaze-teleport-stitch-v0': 0.99,
    'antmaze-medium-explore-v0': 0.99,
    'antmaze-large-explore-v0': 0.99,
    'antmaze-teleport-explore-v0': 0.99,
    
    # Humanoidmaze
    'humanoidmaze-medium-navigate-v0': 0.995,
    'humanoidmaze-large-navigate-v0': 0.995,
    'humanoidmaze-giant-navigate-v0': 0.995,
    'humanoidmaze-medium-stitch-v0': 0.995,
    'humanoidmaze-large-stitch-v0': 0.995,
    'humanoidmaze-giant-stitch-v0': 0.995,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate TSV configuration for RWS-weighted policy training with 5 seeds per agent')
    parser.add_argument('--weights-dir', 
                       default='/scratch/work/yangw4/ogbench/weights/ReachabilityEstimation',
                       help='Path to RWS weights directory')
    parser.add_argument('--output', '-o', 
                       default='rws_weighted_runs.tsv',
                       help='Output TSV file')
    parser.add_argument('--agents', nargs='+', 
                       default=['gcivl', 'gciql', 'hiql'],  # Default to gciql only for RWS-weighted
                       choices=['gcbc', 'gcivl', 'gciql', 'hiql'],
                       help='Agent types to train (default: gciql)')
    parser.add_argument('--num-seeds', type=int,
                       default=5,
                       help='Number of random seeds per configuration (default: 5)')
    parser.add_argument('--weightings', nargs='+',
                       default=['adv'],#['vanilla', 'exponential', 'indicator'],
                       help='Reachability weighting schemes')
    parser.add_argument('--epoch', type=int,
                       default=400000,
                       help='RWS checkpoint epoch to use')
    parser.add_argument('--tasks', nargs='+',
                       help='Specific tasks to include (default: all found)')
    parser.add_argument('--exclude-tasks', nargs='+',
                       help='Tasks to exclude')
    parser.add_argument('--gciql-only', action='store_true',
                       help='Only generate GCIQL configs (most compatible with RWS)')
    return parser.parse_args()

def get_agent_alpha(agent, task):
    """Get the appropriate alpha value for an agent/task combination."""
    if agent not in AGENT_CONFIGS:
        return None
    
    config = AGENT_CONFIGS[agent]
    if config['alpha'] is None:
        return None
    
    # Handle task-specific alpha for GCIQL
    if agent == 'gciql' and isinstance(config['alpha'], dict):
        if 'explore' in task:
            return config['alpha']['explore']
        elif 'humanoidmaze' in task:
            return config['alpha']['humanoidmaze']
        elif 'antmaze' in task:
            return config['alpha']['antmaze']
        else:  # pointmaze
            return config['alpha']['pointmaze']
    
    return config['alpha']

def get_agent_extra_args(agent, task):
    """Get the appropriate extra arguments for an agent/task combination."""
    if agent not in AGENT_CONFIGS:
        return None
    
    config = AGENT_CONFIGS[agent]
    if config['extra_args'] is None:
        return None
    
    # Handle task-specific extra args for HIQL
    if agent == 'hiql' and isinstance(config['extra_args'], dict):
        if 'explore' in task:
            return config['extra_args']['explore']
        elif 'humanoidmaze' in task:
            return config['extra_args']['humanoidmaze']
        else:
            return config['extra_args']['default']
    
    return config['extra_args']

def get_actor_probs(task):
    """Get actor probability settings for a task."""
    if 'stitch' in task:
        return ACTOR_PROBS['stitch']
    elif 'explore' in task:
        return ACTOR_PROBS['explore']
    return {'actor_p_randomgoal': None, 'actor_p_trajgoal': None}

def find_rws_checkpoints(weights_dir: str) -> Dict[str, List[Tuple[str, int]]]:
    """Find all available RWS checkpoints with params_*.pkl files.
    
    Returns:
        Dict mapping task name to list of (subdirectory_name, epoch) tuples
    """
    checkpoints = {}
    weights_path = Path(weights_dir)
    
    if not weights_path.exists():
        print(f"Warning: Weights directory not found: {weights_dir}")
        return checkpoints
    
    # Look for RWS directories
    for rws_dir in weights_path.glob("rws_*_rws"):
        # Extract task name from directory
        match = re.match(r"rws_(.+)_rws", rws_dir.name)
        if not match:
            continue
        
        task = match.group(1)
        checkpoints[task] = []
        
        # Find subdirectories with pattern sd042_s_*.0.*
        for subdir in rws_dir.glob("sd042_s_*.0.*"):
            if not subdir.is_dir():
                continue
            
            # Find params_*.pkl checkpoint files
            epochs = set()
            for ckpt in subdir.glob("params_*.pkl"):
                # Extract epoch number from params_NNNNNN.pkl
                epoch_match = re.search(r'params_(\d+)\.pkl', ckpt.name)
                if epoch_match:
                    epochs.add(int(epoch_match.group(1)))
            
            # Store subdirectory name and available epochs
            subdir_name = subdir.name
            for epoch in sorted(epochs):
                checkpoints[task].append((subdir_name, epoch))
    
    return checkpoints

def generate_config(args):
    """Generate TSV configuration file with 5 seeds per configuration."""
    
    # Find available checkpoints
    print(f"Searching for RWS checkpoints in: {args.weights_dir}")
    checkpoints = find_rws_checkpoints(args.weights_dir)
    
    if not checkpoints:
        print("No RWS checkpoints found!")
        print("Make sure your weights directory contains subdirectories like:")
        print("  rws_antmaze-medium-navigate-v0_rws/sd042_s_12899477.0.20251028_185233/params_400000.pkl")
        return
    
    print(f"Found checkpoints for {len(checkpoints)} tasks")
    
    # Filter tasks if specified
    if args.tasks:
        checkpoints = {k: v for k, v in checkpoints.items() if k in args.tasks}
    
    if args.exclude_tasks:
        checkpoints = {k: v for k, v in checkpoints.items() if k not in args.exclude_tasks}
    
    # Use only GCIQL if specified
    agents = ['gciql'] if args.gciql_only else args.agents
    
    # Generate TSV lines
    lines = []
    header = "# Task\tAgent\tSeed\tAlpha\tDiscount\tRWSSubdir\tRWSEpoch\tReachWeighting\tExtraArgs"
    lines.append(header)
    
    for task in sorted(checkpoints.keys()):
        task_checkpoints = checkpoints[task]
        if not task_checkpoints:
            print(f"Warning: No checkpoints found for {task}")
            continue
        
        # Find checkpoint closest to requested epoch
        best_ckpt = None
        best_diff = float('inf')
        for subdir_name, ckpt_epoch in task_checkpoints:
            diff = abs(ckpt_epoch - args.epoch)
            if diff < best_diff:
                best_diff = diff
                best_ckpt = (subdir_name, ckpt_epoch)
        
        if not best_ckpt:
            print(f"Warning: No suitable checkpoint for {task}")
            continue
        
        subdir_name, ckpt_epoch = best_ckpt
        
        # Get discount factor
        discount = DISCOUNT_FACTORS.get(task, 0.99)
        
        # Get actor probabilities for this task
        actor_probs = get_actor_probs(task)
        
        # Generate lines for each combination
        for agent in agents:
            # Get agent-specific parameters
            alpha = get_agent_alpha(agent, task)
            alpha_str = str(alpha) if alpha is not None else "-"
            
            extra_args = get_agent_extra_args(agent, task)
            extra_args_str = extra_args if extra_args else "-"
            
            for weighting in args.weightings:
                # Generate 5 seeds for each agent/task/weighting combination
                for seed in range(args.num_seeds):
                    # Format: Task Agent Seed Alpha Discount RWSSubdir RWSEpoch ReachWeighting ExtraArgs
                    line = f"{task}\t{agent}\t{seed}\t{alpha_str}\t{discount}\t"
                    line += f"{subdir_name}\t{ckpt_epoch}\t{weighting}\t{extra_args_str}"
                    lines.append(line)
    
    # Write TSV file
    with open(args.output, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    
    print(f"\nGenerated {len(lines)-1} training configurations")
    print(f"Output saved to: {args.output}")
    
    # Print summary
    unique_tasks = len(checkpoints)
    num_agents = len(agents)
    num_weightings = len(args.weightings)
    num_seeds = args.num_seeds
    
    print("\nConfiguration Summary:")
    print(f"  Tasks: {unique_tasks}")
    print(f"  Agents: {agents}")
    print(f"  Weightings: {args.weightings}")
    print(f"  Seeds per config: {num_seeds} (0-{num_seeds-1})")
    print(f"  Total runs: {len(lines)-1}")
    print(f"  Expected: {unique_tasks} × {num_agents} × {num_weightings} × {num_seeds} = {unique_tasks * num_agents * num_weightings * num_seeds}")
    
    # Print example lines
    if len(lines) > 1:
        print("\nFirst few lines of configuration:")
        for line in lines[:min(10, len(lines))]:
            print(f"  {line}")
    
    # Estimate time
    hours_per_run = 24  # Based on SBATCH time limit
    total_hours = (len(lines)-1) * hours_per_run
    print(f"\nEstimated compute time (sequential): {total_hours} hours")
    print(f"With concurrency=6: ~{total_hours/6:.1f} hours")
    print(f"With concurrency=12: ~{total_hours/12:.1f} hours")

def main():
    args = parse_args()
    generate_config(args)

if __name__ == '__main__':
    main()