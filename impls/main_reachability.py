"""
Unified reachability estimator training script - JAX/FLAX VERSION
Supports RWS, TD-RWS, and Expectile Steps agents
Template-based with wandb logging and structured experiment tracking
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

# Import from datasets.py
from utils.datasets import ReachabilityGCDataset, load_maze_trajectories, load_ogbench_trajectories

# Import all agents
from agents.rws import RWSAgent, get_config as get_rws_config
from agents.td_rws import TDRWSAgent, get_config as get_td_rws_config
from agents.expect_rws import ExpectileStepsAgent, get_config as get_expectile_steps_config

from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.flax_utils import save_agent, restore_agent

FLAGS = flags.FLAGS

# Experiment settings
flags.DEFINE_string('run_group', 'ReachabilityRWS', 'Run group.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# Agent selection
flags.DEFINE_enum('agent_type', 'rws', ['rws', 'td_rws', 'expectile_steps'], 'Agent type to use.')

# Dataset settings
flags.DEFINE_enum('dataset_type', 'maze', ['ogbench', 'maze'], 'Source dataset.')
flags.DEFINE_string('dataset_name', None, 'OGBench dataset name.')
flags.DEFINE_enum('dataset_split', 'train', ['train', 'val'], 'Dataset split.')
flags.DEFINE_boolean('compact_ogbench', False, 'Use compact OGBench dataset.')
flags.DEFINE_string('maze_buffer', 'env/A_star_buffer.pkl', 'Path to maze buffer.')

# Training settings
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('steps_per_epoch', 0, 'Steps per epoch (0 = auto).')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('viz_interval', 100000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 500000, 'Saving interval.')

# Model settings
flags.DEFINE_list('hidden_dims', ['256', '256', '256'], 'Hidden layer dimensions.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_float('tau', 0.995, 'Target network soft update rate.')

# Loss settings (RWS)
flags.DEFINE_float('rank_margin', 0.0, 'Rank loss margin.')
flags.DEFINE_float('lambda_cons', 1.0, 'Consistency loss weight (RWS).')

# Loss settings (TD-RWS specific)
flags.DEFINE_float('lambda_cons_pos', 1.0, 'Positive consistency loss weight (TD-RWS).')
flags.DEFINE_float('lambda_cons_unl', 1.0, 'Unlabeled consistency loss weight (TD-RWS).')
flags.DEFINE_float('discount', 0.99, 'Discount factor γ for exponential bounds (TD-RWS).')

# Loss settings (Expectile Steps specific)
flags.DEFINE_float('expectile', 0.7, 'Expectile parameter for step prediction (Expectile Steps).')

# Step-based settings (TD-RWS-Steps and Expectile Steps)
flags.DEFINE_float('h_max', None, 'Maximum step horizon (None = auto: 110 for maze, task_horizon+10 for OGBench).')

# Reachability sampling settings
flags.DEFINE_integer('num_goals_per_state', 4, 'Number of goals per state.')
flags.DEFINE_integer('max_skip_horizon', None, 'Maximum skip horizon (None = 1-step only).')
flags.DEFINE_integer('num_skip_states', 3, 'Number of skip states.')

# Visualization settings
flags.DEFINE_list('viz_dims', ['0', '1'], 'Dimensions to visualize.')
flags.DEFINE_integer('viz_samples', 5000, 'Number of samples for visualization.')
flags.DEFINE_integer('viz_anchors', 9, 'Number of anchor states to visualize.')
flags.DEFINE_enum('viz_scale_mode', 'fixed', ['fixed', 'dynamic'], 
                  'Color scale mode: "fixed" for [0,1] or "dynamic" for [min,max] of data.')

config_flags.DEFINE_config_file('agent', None, lock_config=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def compute_h_max(trajectories: list, dataset_type: str, dataset_name: str = None) -> float:
    """Compute adaptive h_max based on dataset type.
    
    Args:
        trajectories: List of trajectory arrays
        dataset_type: 'maze' or 'ogbench'
        dataset_name: Name of OGBench dataset (if applicable)
    
    Returns:
        h_max: Maximum horizon for step prediction normalization
    """
    if dataset_type == 'maze':
        return 110.0
    else:  # ogbench
        # Compute max trajectory length from the data
        max_traj_len = max(len(traj) for traj in trajectories)
        h_max = max_traj_len + 10.0
        print(f"  Computed h_max from trajectories: {max_traj_len} + 10 = {h_max}")
        return h_max


def visualize_reachability(
    agent,
    agent_type: str,
    dataset: ReachabilityGCDataset,
    step: int,
    save_dir: Path,
    plot_dims: list[int],
    num_anchors: int = 9,
    num_viz_samples: int = 5000,
    h_max: float = 100.0,
    scale_mode: str = 'fixed',
) -> dict:
    """
    Visualize reachability landscapes from multiple anchor states.
    
    Returns:
        Dictionary of visualization metrics for logging.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dims = plot_dims
    if len(dims) != 2:
        raise ValueError("viz_dims must contain exactly two indices for 2D visualization.")

    n_rows = int(np.ceil(np.sqrt(num_anchors)))
    n_cols = int(np.ceil(num_anchors / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if num_anchors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    bounds_min, bounds_max = dataset.state_bounds
    start_states = dataset.start_states
    
    if start_states.shape[0] < num_anchors:
        anchor_indices = np.random.choice(start_states.shape[0], size=num_anchors, replace=True)
    else:
        anchor_indices = np.linspace(0, start_states.shape[0] - 1, num=num_anchors, dtype=int)

    # Get all states and goals, then sample if needed
    _, all_goals, all_states = dataset.get_anchor_state_and_all_goals(anchor_idx=None)
    
    # Sample a subset if dataset is too large
    total_points = all_goals.shape[0]
    if total_points > num_viz_samples:
        sample_indices = np.random.choice(total_points, size=num_viz_samples, replace=False)
        all_goals = all_goals[sample_indices]
        all_states = all_states[sample_indices]

    viz_metrics = {}
    all_reachability_scores = []

    for plot_idx in range(num_anchors):
        ax = axes[plot_idx]
        
        anchor_state = start_states[anchor_indices[plot_idx]]
        
        # Batch evaluation for efficiency
        batch_size = 1024
        reachability_scores = []
        for i in range(0, all_goals.shape[0], batch_size):
            batch_goals = all_goals[i:i + batch_size]
            anchor_batch = np.tile(anchor_state[None, :], (batch_goals.shape[0], 1))
            
            # Get predictions based on agent type
            if agent_type in ['td_rws']:
                # Step-based: convert steps to reachability (1 - steps/h_max)
                steps_jax = agent.predict_reachability(
                    jnp.array(anchor_batch), 
                    jnp.array(batch_goals)
                )
                #scores = 1.0 - np.array(steps_jax).reshape(-1) / h_max
                scores = np.array(steps_jax).reshape(-1)
            else:
                # Reachability-based (RWS)
                scores_jax = agent.predict_reachability(
                    jnp.array(anchor_batch), 
                    jnp.array(batch_goals)
                )
                scores = np.array(scores_jax).reshape(-1)
                #print(scores)
            
            reachability_scores.append(scores)
        
        reachability_scores = np.concatenate(reachability_scores, axis=0)
        all_reachability_scores.extend(reachability_scores)
        goal_coords = all_states[:, dims]
        
        # Determine color scale
        if scale_mode == 'dynamic':
            vmin = np.min(all_reachability_scores)
            vmax = np.max(all_reachability_scores)
        else:
            vmin = 0.0
            vmax = 1.0
        
        scatter = ax.scatter(
            goal_coords[:, 0],
            goal_coords[:, 1],
            c=reachability_scores,
            cmap="RdYlGn",
            s=8,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
        )
        
        anchor_coord = anchor_state[dims]
        ax.scatter(
            anchor_coord[0],
            anchor_coord[1],
            marker="*",
            s=400,
            color="blue",
            edgecolors="white",
            linewidths=2,
            label="Anchor state",
            zorder=10,
        )
        
        cbar = fig.colorbar(scatter, ax=ax)
        if scale_mode == 'dynamic':
            cbar.set_label(f"Reachability [{vmin:.3f}, {vmax:.3f}]", fontsize=10)
        else:
            cbar.set_label("Reachability [0, 1]", fontsize=10)
        
        ax.set_xlim(bounds_min[dims[0]], bounds_max[dims[0]])
        ax.set_ylim(bounds_min[dims[1]], bounds_max[dims[1]])
        ax.set_xlabel(f"Dim {dims[0]}", fontsize=10)
        ax.set_ylabel(f"Dim {dims[1]}", fontsize=10)
        ax.set_title(f"Anchor #{plot_idx + 1}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    for plot_idx in range(num_anchors, len(axes)):
        axes[plot_idx].axis("off")
    
    viz_info = f"({min(total_points, num_viz_samples)}/{total_points} points, scale={scale_mode})"
    fig.suptitle(f"Reachability Landscapes @ Step {step} {viz_info}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"reachability_step_{step:07d}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compute visualization metrics
    viz_metrics['visualization/mean_reachability'] = np.mean(all_reachability_scores)
    viz_metrics['visualization/std_reachability'] = np.std(all_reachability_scores)
    viz_metrics['visualization/min_reachability'] = np.min(all_reachability_scores)
    viz_metrics['visualization/max_reachability'] = np.max(all_reachability_scores)
    viz_metrics['visualization/scale_mode'] = scale_mode
    
    # Log image to wandb
    viz_metrics['visualization/reachability_landscape'] = wandb.Image(str(save_path))
    
    return viz_metrics


def main(_):
    # Set up experiment tracking
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        project='ReachabilityEstimation', 
        group=f'{FLAGS.run_group}_{FLAGS.agent_type.upper()}', 
        name=exp_name
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, 
        wandb.run.project, 
        f'{FLAGS.run_group}_{FLAGS.agent_type}', 
        exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set random seed
    set_seed(FLAGS.seed)

    # Load dataset
    print(f"Loading dataset: {FLAGS.dataset_type}")
    if FLAGS.dataset_type == "ogbench":
        if not FLAGS.dataset_name:
            raise ValueError("--dataset_name must be provided for dataset_type 'ogbench'.")
        print(f"  OGBench dataset: {FLAGS.dataset_name} ({FLAGS.dataset_split})")
        trajectories = load_ogbench_trajectories(
            FLAGS.dataset_name,
            split=FLAGS.dataset_split,
            compact_dataset=FLAGS.compact_ogbench,
        )
    else:
        buffer_path = Path(FLAGS.maze_buffer)
        print(f"  Maze buffer: {buffer_path}")
        trajectories = load_maze_trajectories(buffer_path)

    # Compute adaptive h_max if not specified
    if FLAGS.h_max is None:
        h_max = compute_h_max(trajectories, FLAGS.dataset_type, FLAGS.dataset_name)
        print(f"Using adaptive h_max: {h_max}")
    else:
        h_max = FLAGS.h_max
        print(f"Using specified h_max: {h_max}")

    # Create ReachabilityGCDataset
    dataset = ReachabilityGCDataset(trajectories=trajectories)
    print(f"Dataset size: {len(dataset)} transitions")

    # Parse hidden dimensions
    hidden_dims = [int(d) for d in FLAGS.hidden_dims]

    # Create agent config based on agent type
    print(f"\nInitializing {FLAGS.agent_type.upper()} agent...")
    if FLAGS.agent_type == 'rws':
        config = get_rws_config()
        config['lambda_cons'] = FLAGS.lambda_cons
        AgentClass = RWSAgent
        use_policy_batch = False
    elif FLAGS.agent_type == 'td_rws':
        config = get_td_rws_config()
        config['lambda_cons_pos'] = FLAGS.lambda_cons_pos
        config['lambda_cons_unl'] = FLAGS.lambda_cons_unl
        config['h_max'] = h_max
        AgentClass = TDRWSAgent
        use_policy_batch = False
    elif FLAGS.agent_type == 'expectile_steps':
        config = get_expectile_steps_config()
        config['expectile'] = FLAGS.expectile
        config['h_max'] = h_max
        AgentClass = ExpectileStepsAgent
        use_policy_batch = True  # ExpectileSteps uses GCIVL-style policy batch
    else:
        raise ValueError(f"Unknown agent type: {FLAGS.agent_type}")
    
    # Common config settings
    config['lr'] = FLAGS.lr
    config['tau'] = FLAGS.tau
    config['value_hidden_dims'] = tuple(hidden_dims)
    config['batch_size'] = FLAGS.batch_size
    #config['encoder'] = None
    config['frame_stack'] = None
    config['discount'] = FLAGS.discount
    
    # Additional config for reachability agents (RWS/TD-RWS)
    if not use_policy_batch:
        config['rank_margin'] = FLAGS.rank_margin
        config['num_goals_per_state'] = FLAGS.num_goals_per_state
        config['num_skip_states'] = FLAGS.num_skip_states
        config['max_skip_horizon'] = FLAGS.max_skip_horizon

    # Get example observations for agent initialization
    sample_batch = dataset.sample_batch(
        batch_size=2,
        num_goals_per_state=FLAGS.num_goals_per_state,
        max_skip_horizon=FLAGS.max_skip_horizon,
        num_skip_states=FLAGS.num_skip_states,
    )
    
    # Select appropriate batch for initialization
    if use_policy_batch:
        ex_batch = sample_batch['policy']
        ex_observations = ex_batch['observations']
    else:
        ex_batch = sample_batch['reachability']
        ex_observations = ex_batch['states']
    
    ex_actions = np.zeros((2, 1))  # Dummy actions (not used)

    print(config)
   
    # Initialize agent
    agent = AgentClass.create(
        seed=FLAGS.seed,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )

    # Restore agent if specified
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        print(f"Restored agent from {FLAGS.restore_path}, epoch {FLAGS.restore_epoch}")

    # Calculate steps per epoch if not specified
    steps_per_epoch = FLAGS.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(len(dataset) // FLAGS.batch_size, 1)

    # Log configuration
    print("\nTraining configuration:", FLAGS.discount)
    config_summary = {
        "agent_type": FLAGS.agent_type,
        "dataset_type": FLAGS.dataset_type,
        "dataset_size": len(dataset),
        "train_steps": FLAGS.train_steps,
        "batch_size": FLAGS.batch_size,
        "lr": FLAGS.lr,
        "tau": FLAGS.tau,
        "hidden_dims": hidden_dims,
        "h_max": h_max,
        "use_policy_batch": use_policy_batch,
    }
    
    if FLAGS.agent_type == 'rws':
        config_summary.update({
            'lambda_cons': FLAGS.lambda_cons,
            'rank_margin': FLAGS.rank_margin,
            'discount': FLAGS.discount,
            'num_goals_per_state': FLAGS.num_goals_per_state,
            'max_skip_horizon': FLAGS.max_skip_horizon,
            'num_skip_states': FLAGS.num_skip_states,
        })
    elif FLAGS.agent_type == 'td_rws':
        config_summary.update({
            'lambda_cons_pos': FLAGS.lambda_cons_pos,
            'lambda_cons_unl': FLAGS.lambda_cons_unl,
            'discount': FLAGS.discount,
            'num_goals_per_state': FLAGS.num_goals_per_state,
            'max_skip_horizon': FLAGS.max_skip_horizon,
            'num_skip_states': FLAGS.num_skip_states,
        })
    elif FLAGS.agent_type == 'expectile_steps':
        config_summary.update({
            'expectile': FLAGS.expectile,
        })
    
    print(json.dumps(config_summary, indent=2))
    wandb.config.update(config_summary, allow_val_change=True)

    # Set up loggers
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    viz_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'visualization.csv'))

    # Training loop
    print(f"\nStarting training for {FLAGS.train_steps} steps...")
    first_time = time.time()
    last_time = time.time()
    
    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample batch
        batch_dict = dataset.sample_batch(
            FLAGS.batch_size,
            num_goals_per_state=FLAGS.num_goals_per_state,
            max_skip_horizon=FLAGS.max_skip_horizon,
            num_skip_states=FLAGS.num_skip_states,
        )
        
        # Select appropriate batch based on agent type
        if use_policy_batch:
            batch_np = batch_dict['policy']  # GCIVL-style keys for ExpectileSteps
        else:
            batch_np = batch_dict['reachability']  # Reachability keys for RWS/TD-RWS
        
        # Convert to JAX
        batch = {k: jnp.array(v) for k, v in batch_np.items()}

        # Update agent
        agent, update_info = agent.update(batch)

        # Log training metrics
        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/step_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics['time/steps_per_second'] = FLAGS.log_interval / (time.time() - last_time)
            last_time = time.time()
            
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        # Visualize reachability landscapes
        if step == 1 or step % FLAGS.viz_interval == 0:
            print(f"\nGenerating visualization at step {step}...")
            viz_dir = Path(FLAGS.save_dir) / 'visualizations'
            viz_dims = [int(d) for d in FLAGS.viz_dims]
            
            viz_metrics = visualize_reachability(
                agent=agent,
                agent_type=FLAGS.agent_type,
                dataset=dataset,
                step=step,
                save_dir=viz_dir,
                plot_dims=viz_dims,
                num_anchors=FLAGS.viz_anchors,
                num_viz_samples=FLAGS.viz_samples,
                h_max=h_max,
                scale_mode=FLAGS.viz_scale_mode,
            )
            
            wandb.log(viz_metrics, step=step)
            viz_logger.log(viz_metrics, step=step)
            print(f"Visualization saved. Mean reachability: {viz_metrics['visualization/mean_reachability']:.3f}")

        # Save checkpoint
        if step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)
            print(f"\nCheckpoint saved at step {step}{FLAGS.save_dir}")

    # Final save
    save_agent(agent, FLAGS.save_dir, FLAGS.train_steps)
    print(f"\nTraining complete! Final checkpoint saved.")
    
    train_logger.close()
    viz_logger.close()


if __name__ == '__main__':
    app.run(main)


"""
Example usage:

# ============================================================
# RWS Agent
# ============================================================

# Train RWS agent on Maze
python main_reachability.py \
    --agent_type=rws \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons=1.0 \
    --num_skip_states=50 \
    --run_group=Maze_RWS

# Train RWS on OGBench AntMaze
python main_reachability.py \
    --agent_type=rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons=1.0 \
    --num_skip_states=50 \
    --run_group=AntMaze_RWS

# ============================================================
# TD-RWS Agent (Step-based)
# ============================================================

# Train TD-RWS agent on Maze (h_max auto-computed as 110)
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons_pos=1.0 \
    --lambda_cons_unl=1.0 \
    --discount=0.99 \
    --num_skip_states=50 \
    --run_group=Maze_TDRWS \
    --viz_interval=25000

# Train TD-RWS on OGBench (h_max auto-computed from max trajectory length)
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-giant-stitch-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons_pos=1.0 \
    --lambda_cons_unl=1.0 \
    --discount=0.99 \
    --num_skip_states=50 \
    --run_group=AntMaze_TDRWS
    --viz_interval=25000

# Train TD-RWS with custom h_max
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=maze \
    --h_max=200.0 \
    --batch_size=128 \
    --run_group=Maze_TDRWS_H200

# ============================================================
# ExpectileSteps Agent
# ============================================================

# Train ExpectileSteps on Maze (h_max=110 auto)
python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --expectile=0.7 \
    --lr=3e-4 \
    --tau=0.005 \
    --viz_scale_mode=dynamic \
    --run_group=Maze_ExpectileSteps

# Train ExpectileSteps on OGBench (h_max auto from data)
python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --expectile=0.7 \
    --viz_interval=50000 \
    --viz_scale_mode=dynamic \
    --run_group=AntMaze_ExpectileSteps

# Compare different expectile values on Maze
python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=maze \
    --expectile=0.5 \
    --run_group=Maze_Expectile05

python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=maze \
    --expectile=0.7 \
    --run_group=Maze_Expectile07

python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=maze \
    --expectile=0.9 \
    --run_group=Maze_Expectile09

# ============================================================
# Compare all three agents on same dataset
# ============================================================

# RWS
python main_reachability.py \
    --agent_type=rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --batch_size=128 \
    --run_group=Compare_RWS

# TD-RWS
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --batch_size=128 \
    --run_group=Compare_TDRWS

# ExpectileSteps
python main_reachability.py \
    --agent_type=expectile_steps \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --batch_size=128 \
    --expectile=0.7 \
    --run_group=Compare_ExpectileSteps
"""

"""
Example usage:

# Train RWS agent on Maze
python main_reachability.py \
    --agent_type=rws \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons=1.0 \
    --num_skip_states=50 \
    --run_group=Maze

# Train TD-RWS agent on Maze
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons_pos=1.0 \
    --lambda_cons_unl=1.0 \
    --discount=0.99 \
    --num_skip_states=50 \
    --run_group=Maze

# Train TD-RWS on OGBench AntMaze
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-giant-stitch-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons_pos=1.0 \
    --lambda_cons_unl=1.0 \
    --discount=0.99 \
    --num_skip_states=50 \
    --run_group=AntMaze

# Train TD-RWS-Steps (step-based cost-to-go) on Maze
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --lambda_cons_pos=1.0 \
    --lambda_cons_unl=1.0 \
    --h_max=2000.0 \
    --num_skip_states=50 \
    --run_group=Maze_Steps

# Compare both agents with custom visualization
python main_reachability.py \
    --agent_type=td_rws \
    --dataset_type=ogbench \
    --dataset_name=antmaze-large-explore-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --h_max=2000.0 \
    --viz_interval=50000 \
    --viz_dims=0,1 \
    --viz_samples=10000 \
    --viz_anchors=16 \
    --num_skip_states=20 
"""