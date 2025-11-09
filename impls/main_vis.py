"""
Training script for goal-conditioned offline RL agents with value function visualization.
Supports GCIQL, GCIVL, QRL, HIQL, RWS, and OTA agents with reachability-style visualization.
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

# Import datasets
from utils.datasets import Dataset, GCDataset, HGCDataset, ReachabilityGCDataset, load_maze_trajectories, load_ogbench_trajectories
from utils.env_utils import make_env_and_datasets

# Import agents
from agents import agents

from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.flax_utils import save_agent, restore_agent

FLAGS = flags.FLAGS

# Experiment settings
flags.DEFINE_string('run_group', 'GCRL_Vis', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# Dataset settings
flags.DEFINE_enum('dataset_type', 'ogbench', ['ogbench', 'maze'], 'Source dataset.')
flags.DEFINE_string('dataset_name', None, 'OGBench dataset name (for dataset_type=ogbench).')
flags.DEFINE_string('env_name', None, 'Environment name (alternative to dataset_name).')
flags.DEFINE_string('maze_buffer', 'env/A_star_buffer.pkl', 'Path to maze buffer (for dataset_type=maze).')

# Training settings
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('viz_interval', 100000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 500000, 'Saving interval.')

# Visualization settings
flags.DEFINE_list('viz_dims', ['0', '1'], 'Dimensions to visualize.')
flags.DEFINE_integer('viz_samples', 5000, 'Number of samples for visualization.')
flags.DEFINE_integer('viz_anchors', 9, 'Number of anchor states to visualize.')
flags.DEFINE_enum('viz_scale_mode', 'dynamic', ['fixed', 'dynamic'], 
                  'Color scale mode (kept for backward compatibility, always uses adaptive/dynamic scaling based on actual data min/max).')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_value_prediction(agent, states, goals, agent_name):
    """
    Get value predictions from different agent types.
    
    Args:
        agent: The agent instance
        states: Batch of states [batch_size, state_dim]
        goals: Batch of goals [batch_size, state_dim]
        agent_name: Name of the agent ('gciql', 'gcivl', 'qrl', 'hiql', 'rws', 'ota', etc.)
    
    Returns:
        values: Predicted values [batch_size]
    """
    if agent_name in ['gciql', 'gcivl', 'hiql']:
        # For IQL-based agents, use the value network V(s, g)
        # These agents have ensemble value functions, so we average them
        if agent_name in ['gcivl', 'hiql']:
            v1, v2 = agent.network.select('value')(states, goals)
            values = (v1 + v2) / 2
        else:  # gciql
            values = agent.network.select('value')(states, goals)
        
        # Convert to numpy and return
        return np.array(values).reshape(-1)
    
    elif agent_name == 'qrl':
        # For QRL, use -d(s, g) where d is the quasimetric distance
        distances = agent.network.select('value')(states, goals)
        values = -np.array(distances).reshape(-1)
        return values
    
    elif agent_name == 'rws':
        # For RWS, use predict_reachability method
        scores = agent.predict_reachability(states, goals)
        return np.array(scores).reshape(-1)
    
    elif agent_name == 'ota':
        # For OTA (Option-aware Temporally Abstracted), use the low-level value function
        # OTA has ensemble low_value functions (v1, v2), so we average them
        v1, v2 = agent.network.select('high_value')(states, goals)
        values = (v1 + v2) / 2
        return np.array(values).reshape(-1)
    
    else:
        raise ValueError(f'Unsupported agent type for visualization: {agent_name}')


def visualize_value_function(
    agent,
    agent_name: str,
    dataset: ReachabilityGCDataset,
    step: int,
    save_dir: Path,
    plot_dims: list[int],
    num_anchors: int = 9,
    num_viz_samples: int = 5000,
    scale_mode: str = 'fixed',  # Kept for backward compatibility but ignored
) -> dict:
    """
    Visualize value function landscapes from multiple anchor states.
    Similar to reachability visualization but for goal-conditioned value functions.
    
    Note: scale_mode parameter is ignored - always uses adaptive scaling based on actual min/max values.
    
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

    # Get all states and goals
    _, all_goals, all_states = dataset.get_anchor_state_and_all_goals(anchor_idx=None)
    
    # Sample a subset if dataset is too large
    total_points = all_goals.shape[0]
    if total_points > num_viz_samples:
        sample_indices = np.random.choice(total_points, size=num_viz_samples, replace=False)
        all_goals = all_goals[sample_indices]
        all_states = all_states[sample_indices]

    viz_metrics = {}
    all_value_scores = []

    for plot_idx in range(num_anchors):
        ax = axes[plot_idx]
        
        anchor_state = start_states[anchor_indices[plot_idx]]
        
        # Batch evaluation for efficiency
        batch_size = 1024
        value_scores = []
        for i in range(0, all_goals.shape[0], batch_size):
            batch_goals = all_goals[i:i + batch_size]
            anchor_batch = np.tile(anchor_state[None, :], (batch_goals.shape[0], 1))
            
            # Get value predictions
            scores = get_value_prediction(
                agent, 
                jnp.array(anchor_batch), 
                jnp.array(batch_goals),
                agent_name
            )
            
            value_scores.append(scores)
        
        value_scores = np.concatenate(value_scores, axis=0)
        all_value_scores.extend(value_scores)
        goal_coords = all_states[:, dims]
        
        # Always use dynamic scale - no normalization, just show actual values
        vmin = np.min(value_scores)
        vmax = np.max(value_scores)
        
        scatter = ax.scatter(
            goal_coords[:, 0],
            goal_coords[:, 1],
            c=value_scores,
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
        cbar.set_label(f"Value [{vmin:.3f}, {vmax:.3f}]", fontsize=10)
        
        ax.set_xlabel(f"State Dim {dims[0]}", fontsize=12)
        ax.set_ylabel(f"State Dim {dims[1]}", fontsize=12)
        ax.set_title(f"Anchor #{plot_idx + 1}", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_anchors, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    viz_path = save_dir / f"value_viz_step_{step:08d}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Compute metrics
    viz_metrics['visualization/mean_value'] = float(np.mean(all_value_scores))
    viz_metrics['visualization/std_value'] = float(np.std(all_value_scores))
    viz_metrics['visualization/min_value'] = float(np.min(all_value_scores))
    viz_metrics['visualization/max_value'] = float(np.max(all_value_scores))
    
    # Log the image to wandb
    viz_metrics['visualization/value_landscape'] = wandb.Image(str(viz_path))
    
    return viz_metrics


def main(_):
    # Set up logger
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='GCRL_Visualization', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set random seed
    set_seed(FLAGS.seed)
    
    # Load configuration
    config = FLAGS.agent
    agent_name = config['agent_name']
    
    # Add default sampling parameters for RWS if not present
    if agent_name == 'rws':
        # Convert to dict if it's a FrozenDict
        if hasattr(config, 'to_dict'):
            config = dict(config.to_dict())
        else:
            config = dict(config)
        config.setdefault('num_goals_per_state', 4)
        config.setdefault('max_skip_horizon', None)
        config.setdefault('num_skip_states', 3)
    
    print(f"\n{'='*60}")
    print(f"Training {agent_name.upper()} agent with visualization")
    print(f"{'='*60}\n")

    # Import Dataset class (needed for both maze and OGBench)
    from utils.datasets import Dataset as BaseDataset

    # Set up dataset
    if FLAGS.dataset_type == 'maze':
        print(f"Loading maze dataset from: {FLAGS.maze_buffer}")
        trajectories = load_maze_trajectories(FLAGS.maze_buffer)
        
        # Build dataset with terminals marked at trajectory ends
        all_obs = []
        all_next_obs = []
        all_actions = []
        all_rewards = []
        all_masks = []
        all_terminals = []
        
        for traj in trajectories:
            traj_len = len(traj) - 1
            all_obs.append(traj[:-1])
            all_next_obs.append(traj[1:])
            all_actions.append(np.zeros((traj_len, 2)))  # dummy actions
            all_rewards.append(np.zeros(traj_len))
            all_masks.append(np.ones(traj_len))
            
            # Mark terminal at end of trajectory
            terminals = np.zeros(traj_len)
            terminals[-1] = 1.0
            all_terminals.append(terminals)
        
        dataset_dict = {
            'observations': np.concatenate(all_obs, axis=0),
            'next_observations': np.concatenate(all_next_obs, axis=0),
            'actions': np.concatenate(all_actions, axis=0),
            'rewards': np.concatenate(all_rewards, axis=0),
            'masks': np.concatenate(all_masks, axis=0),
            'terminals': np.concatenate(all_terminals, axis=0),
        }
        base_dataset = BaseDataset.create(**dataset_dict)
        
        # Create a simple dummy env for maze
        class DummyEnv:
            def __init__(self):
                self.action_space = type('obj', (object,), {'n': 4, 'shape': (2,)})()
        env = DummyEnv()
        
    else:  # ogbench
        env_name = FLAGS.env_name if FLAGS.env_name is not None else FLAGS.dataset_name
        print(f"Loading OGBench dataset: {env_name}")
        env, train_dataset_dict, val_dataset_dict = make_env_and_datasets(
            env_name, 
            frame_stack=config.get('frame_stack', None)
        )
        base_dataset = BaseDataset.create(**train_dataset_dict)
    
    # Create goal-conditioned dataset based on config
    dataset_class_name = config.get('dataset_class', 'GCDataset')
    
    # Special handling for RWS agent - it uses ReachabilityGCDataset directly with trajectories
    if agent_name == 'rws':
        print("Using ReachabilityGCDataset for RWS agent")
        if FLAGS.dataset_type == 'maze':
            # ReachabilityGCDataset only takes trajectories as keyword argument
            train_dataset = ReachabilityGCDataset(trajectories=trajectories)
        else:  # ogbench
            from utils.datasets import load_ogbench_trajectories
            train_trajectories = load_ogbench_trajectories(env_name)
            train_dataset = ReachabilityGCDataset(trajectories=train_trajectories)
    else:
        # Standard GC agents use wrapped datasets
        if dataset_class_name == 'GCDataset':
            dataset_class = GCDataset
        elif dataset_class_name == 'HGCDataset':
            dataset_class = HGCDataset
        elif dataset_class_name == 'ReachabilityDataset':
            dataset_class = ReachabilityGCDataset
        else:
            dataset_class = GCDataset
        
        train_dataset = dataset_class(base_dataset, config)
    
    # For visualization, we need a ReachabilityGCDataset
    if FLAGS.dataset_type == 'maze':
        # ReachabilityGCDataset only takes trajectories parameter
        viz_dataset = ReachabilityGCDataset(trajectories=trajectories)
    else:
        # For OGBench, create visualization dataset
        from utils.datasets import load_ogbench_trajectories
        viz_trajectories = load_ogbench_trajectories(env_name)
        viz_dataset = ReachabilityGCDataset(trajectories=viz_trajectories)

    # Print dataset info
    if agent_name == 'rws':
        print(f"Dataset created for RWS agent")
    else:
        print(f"Dataset created with {len(base_dataset)} transitions")
    print(f"Visualization dataset created")

    # Initialize agent
    if agent_name == 'rws':
        # RWS needs example observations from its specific batch format
        sample_batch_dict = train_dataset.sample_batch(
            batch_size=2,
            num_goals_per_state=config.get('num_goals_per_state', 4),
            max_skip_horizon=config.get('max_skip_horizon', None),
            num_skip_states=config.get('num_skip_states', 3),
        )
        ex_batch = sample_batch_dict['reachability']
        example_observations = ex_batch['states']
        example_actions = np.zeros((2, 1))  # Dummy actions
    else:
        # Standard GC agents
        example_batch = train_dataset.sample(1)
        if config.get('discrete', False):
            example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)
        example_observations = example_batch['observations']
        example_actions = example_batch['actions']

    agent_class = agents[agent_name]
    agent_kwargs = {}
    
    print(f"\nInitializing {agent_name} agent...")
    agent = agent_class.create(
        FLAGS.seed,
        example_observations,
        example_actions,
        config,
        **agent_kwargs,
    )

    # Restore agent if needed
    if FLAGS.restore_path is not None:
        print(f"Restoring agent from {FLAGS.restore_path} at epoch {FLAGS.restore_epoch}")
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Training loop
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    viz_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'viz.csv'))
    
    print(f"\nStarting training for {FLAGS.train_steps} steps...")
    print(f"Logging every {FLAGS.log_interval} steps")
    print(f"Visualizing every {FLAGS.viz_interval} steps")
    print(f"Saving every {FLAGS.save_interval} steps\n")
    
    first_time = time.time()
    last_time = time.time()
    
    for step in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent
        if agent_name == 'rws':
            # RWS uses sample_batch with specific parameters
            batch_dict = train_dataset.sample_batch(
                batch_size=config['batch_size'],
                num_goals_per_state=config.get('num_goals_per_state', 4),
                max_skip_horizon=config.get('max_skip_horizon', None),
                num_skip_states=config.get('num_skip_states', 3),
            )
            batch_np = batch_dict['reachability']
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
        else:
            # Standard GC agents use sample method
            batch = train_dataset.sample(config['batch_size'])
        
        agent, update_info = agent.update(batch)

        # Log metrics
        if step % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/step_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            
            wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)

        # Visualize value function
        if step == 1 or step % FLAGS.viz_interval == 0:
            print(f"\nGenerating visualization at step {step}...")
            
            plot_dims = [int(d) for d in FLAGS.viz_dims]
            viz_metrics = visualize_value_function(
                agent=agent,
                agent_name=agent_name,
                dataset=viz_dataset,
                step=step,
                save_dir=Path(FLAGS.save_dir),
                plot_dims=plot_dims,
                num_anchors=FLAGS.viz_anchors,
                num_viz_samples=FLAGS.viz_samples,
                scale_mode=FLAGS.viz_scale_mode,
            )
            
            wandb.log(viz_metrics, step=step)
            viz_logger.log(viz_metrics, step=step)
            print(f"Visualization saved. Mean value: {viz_metrics['visualization/mean_value']:.3f}")

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

# ============================================================
# Train GCIQL on OGBench with visualization
# ============================================================

python main_vis.py \
    --dataset_type=ogbench \
    --env_name=antmaze-large-navigate-v0 \
    --train_steps=1000000 \
    --agent=agents/gciql.py \
    --agent.alpha=0.003 \
    --agent.actor_loss=ddpgbc \
    --viz_interval=50000 \
    --run_group=GCIQL_AntMaze

# ============================================================
# Train OTA on OGBench with visualization
# ============================================================

python main_vis.py \
    --dataset_type=ogbench \
    --env_name=antmaze-large-navigate-v0 \
    --train_steps=1000000 \
    --agent=agents/ota.py \
    --agent.low_alpha=3.0 \
    --agent.high_alpha=3.0 \
    --agent.subgoal_steps=25 \
    --agent.rep_dim=10 \
    --viz_interval=50000 \
    --run_group=OTA_AntMaze

# ============================================================
# Train OTA on maze with visualization
# ============================================================

python main_vis.py \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --train_steps=1000000 \
    --agent=agents/ota.py \
    --agent.low_alpha=3.0 \
    --agent.high_alpha=3.0 \
    --agent.subgoal_steps=25 \
    --agent.rep_dim=10 \
    --viz_interval=25000 \
    --run_group=OTA_Maze

# ============================================================
# Train GCIVL with visualization
# ============================================================

python main_vis.py \
    --dataset_type=ogbench \
    --env_name=pointmaze-large-stitch-v0 \
    --train_steps=1000000 \
    --agent=agents/gcivl.py \
    --agent.alpha=10.0 \
    --viz_interval=50000 \
    --run_group=GCIVL_PointMaze

# ============================================================
# Train QRL with visualization
# ============================================================

python main_vis.py \
    --dataset_type=ogbench \
    --env_name=antmaze-large-navigate-v0 \
    --train_steps=1000000 \
    --agent=agents/qrl.py \
    --agent.alpha=0.003 \
    --agent.actor_loss=ddpgbc \
    --viz_interval=50000 \
    --run_group=QRL_AntMaze

# ============================================================
# Train RWS with visualization
# ============================================================

python main_vis.py \
    --dataset_type=maze \
    --maze_buffer=env/A_star_buffer.pkl \
    --train_steps=1000000 \
    --agent=agents/rws.py \
    --viz_interval=25000 \
    --run_group=RWS_Maze

# ============================================================
# Train HIQL with visualization
# ============================================================

python main_vis.py \
    --dataset_type=ogbench \
    --env_name=antmaze-large-navigate-v0 \
    --train_steps=1000000 \
    --agent=agents/hiql.py \
    --agent.low_alpha=3.0 \
    --agent.high_alpha=3.0 \
    --agent.subgoal_steps=25 \
    --agent.rep_dim=10 \
    --viz_interval=50000 \
    --run_group=HIQL_AntMaze
"""