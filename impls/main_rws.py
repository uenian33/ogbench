"""
Reachability estimator training script - JAX/FLAX VERSION using RWSAgent
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

# Import RWSAgent from rws.py
from agents.rws import RWSAgent, get_config

from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.flax_utils import save_agent, restore_agent

FLAGS = flags.FLAGS

# Experiment settings
flags.DEFINE_string('run_group', 'ReachabilityRWS', 'Run group.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

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

# Loss settings
flags.DEFINE_float('rank_margin', 0.0, 'Rank loss margin.')
flags.DEFINE_float('lambda_cons', 1.0, 'Consistency loss weight.')

# Reachability sampling settings
flags.DEFINE_integer('num_goals_per_state', 4, 'Number of goals per state.')
flags.DEFINE_integer('max_skip_horizon', None, 'Maximum skip horizon (None = 1-step only).')
flags.DEFINE_integer('num_skip_states', 3, 'Number of skip states.')

# Visualization settings
flags.DEFINE_list('viz_dims', ['0', '1'], 'Dimensions to visualize.')
flags.DEFINE_integer('viz_samples', 5000, 'Number of samples for visualization.')
flags.DEFINE_integer('viz_anchors', 9, 'Number of anchor states to visualize.')

config_flags.DEFINE_config_file('agent', 'agents/rws.py', lock_config=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def visualize_reachability(
    agent: RWSAgent,
    dataset: ReachabilityGCDataset,
    step: int,
    save_dir: Path,
    plot_dims: list[int],
    num_anchors: int = 9,
    num_viz_samples: int = 5000,
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
            scores_jax = agent.predict_reachability(
                jnp.array(anchor_batch), 
                jnp.array(batch_goals)
            )
            scores = np.array(scores_jax).reshape(-1)
            reachability_scores.append(scores)
        
        reachability_scores = np.concatenate(reachability_scores, axis=0)
        all_reachability_scores.extend(reachability_scores)
        goal_coords = all_states[:, dims]
        
        scatter = ax.scatter(
            goal_coords[:, 0],
            goal_coords[:, 1],
            c=reachability_scores,
            cmap="RdYlGn",
            s=8,
            alpha=0.6,
            vmin=0.0,
            vmax=1.0,
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
        cbar.set_label("Reachability", fontsize=10)
        
        ax.set_xlim(bounds_min[dims[0]], bounds_max[dims[0]])
        ax.set_ylim(bounds_min[dims[1]], bounds_max[dims[1]])
        ax.set_xlabel(f"Dim {dims[0]}", fontsize=10)
        ax.set_ylabel(f"Dim {dims[1]}", fontsize=10)
        ax.set_title(f"Anchor #{plot_idx + 1}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    for plot_idx in range(num_anchors, len(axes)):
        axes[plot_idx].axis("off")
    
    viz_info = f"({min(total_points, num_viz_samples)}/{total_points} points)"
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
    
    # Log image to wandb
    viz_metrics['visualization/reachability_landscape'] = wandb.Image(str(save_path))
    
    return viz_metrics


def main(_):
    # Set up experiment tracking
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='ReachabilityEstimation', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
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

    # Create ReachabilityGCDataset
    dataset = ReachabilityGCDataset(trajectories=trajectories)
    print(f"Dataset size: {len(dataset)} transitions")

    # Parse hidden dimensions
    hidden_dims = [int(d) for d in FLAGS.hidden_dims]

    # Create agent config
    config = get_config()
    config['lr'] = FLAGS.lr
    config['tau'] = FLAGS.tau
    config['rank_margin'] = FLAGS.rank_margin
    config['lambda_cons'] = FLAGS.lambda_cons
    config['value_hidden_dims'] = tuple(hidden_dims)
    config['batch_size'] = FLAGS.batch_size
    config['num_goals_per_state'] = FLAGS.num_goals_per_state
    config['num_skip_states'] = FLAGS.num_skip_states
    config['max_skip_horizon'] = FLAGS.max_skip_horizon
    config['encoder'] = None
    config['frame_stack'] = None

    # Get example observations for agent initialization
    sample_batch = dataset.sample_batch(
        batch_size=2,
        num_goals_per_state=FLAGS.num_goals_per_state,
        max_skip_horizon=FLAGS.max_skip_horizon,
        num_skip_states=FLAGS.num_skip_states,
    )
    ex_observations = sample_batch['reachability']['states']
    ex_actions = np.zeros((2, 1))  # Dummy actions (not used in RWS)

    # Initialize agent
    agent = RWSAgent.create(
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
    print("\nTraining configuration:")
    config_summary = {
        "dataset_size": len(dataset),
        "hidden_dims_parsed": hidden_dims,  # Use different key to avoid conflict
    }
    print(json.dumps({
        "dataset_type": FLAGS.dataset_type,
        "dataset_size": len(dataset),
        "train_steps": FLAGS.train_steps,
        "batch_size": FLAGS.batch_size,
        "lr": FLAGS.lr,
        "tau": FLAGS.tau,
        "rank_margin": FLAGS.rank_margin,
        "lambda_cons": FLAGS.lambda_cons,
        "hidden_dims": hidden_dims,
        "num_goals_per_state": FLAGS.num_goals_per_state,
        "max_skip_horizon": FLAGS.max_skip_horizon,
        "num_skip_states": FLAGS.num_skip_states,
    }, indent=2))
    wandb.config.update(config_summary, allow_val_change=True)

    # Set up loggers
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    viz_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'visualization.csv'))

    # Training loop
    print(f"\nStarting training for {FLAGS.train_steps} steps...")
    first_time = time.time()
    last_time = time.time()
    
    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Sample batch and convert to JAX
        batch_dict = dataset.sample_batch(
            FLAGS.batch_size,
            num_goals_per_state=FLAGS.num_goals_per_state,
            max_skip_horizon=FLAGS.max_skip_horizon,
            num_skip_states=FLAGS.num_skip_states,
        )
        batch_np = batch_dict['reachability']
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
                dataset=dataset,
                step=step,
                save_dir=viz_dir,
                plot_dims=viz_dims,
                num_anchors=FLAGS.viz_anchors,
                num_viz_samples=FLAGS.viz_samples,
            )
            
            wandb.log(viz_metrics, step=step)
            viz_logger.log(viz_metrics, step=step)
            print(f"Visualization saved. Mean reachability: {viz_metrics['visualization/mean_reachability']:.3f}")

        # Save checkpoint
        if step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)
            print(f"\nCheckpoint saved at step {step}")

    # Final save
    save_agent(agent, FLAGS.save_dir, FLAGS.train_steps)
    print(f"\nTraining complete! Final checkpoint saved.")
    
    train_logger.close()
    viz_logger.close()


if __name__ == '__main__':
    app.run(main)


"""
Example usage:

# Maze environment
python test.py \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --num_skip_states=50 \
    --run_group=Maze

# OGBench AntMaze
python test.py \
    --dataset_type=ogbench \
    --dataset_name=antmaze-giant-stitch-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --num_skip_states=50 \
    --run_group=AntMaze

# With custom visualization settings
python test.py \
    --dataset_type=ogbench \
    --dataset_name=antmaze-medium-navigate-v0 \
    --hidden_dims=256,256,256 \
    --train_steps=1000000 \
    --batch_size=128 \
    --viz_interval=50000 \
    --viz_dims=0,1 \
    --viz_samples=10000 \
    --viz_anchors=16
"""