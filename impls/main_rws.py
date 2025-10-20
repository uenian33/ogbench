import json
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import ReachabilityGCDataset, load_maze_trajectories, load_ogbench_trajectories
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from jax import device_put

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'RWS_Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('viz_interval', 10000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

flags.DEFINE_string('dataset_type', "ogbench", 'dataset type.')
flags.DEFINE_string('dataset_name', "antmaze-teleport-stitch-v0", 'dataset type.')
flags.DEFINE_string('dataset_split', 'train', 'Dataset split to load when using ogbench datasets.')
flags.DEFINE_bool('compact_ogbench', True, 'Load compact ogbench datasets (observations only).')
flags.DEFINE_string('maze_buffer', None, 'Path to offline maze replay buffer (pickle).')

flags.DEFINE_integer('num_goals_per_state', 4, 'Number of goals per state.')
flags.DEFINE_integer('max_skip_horizon', None, '`Maximum skip horizon for RWS.')
flags.DEFINE_integer('num_skip_states', 10, '`Number of intermediate states for RWS.')

flags.DEFINE_integer('num_viz_anchors', 9, 'Number of anchor states for visualization.')
flags.DEFINE_integer('num_viz_points', 5000, 'Number of points to visualize.')

config_flags.DEFINE_config_file('agent', 'agents/rws.py', lock_config=False)


def _prepare_reachability_batch(batch_np):
    """Move the reachability batch to device and drop unused keys."""
    return {
        k: device_put(v)
        for k, v in batch_np.items()
        if k in ("states", "skip_states", "positive_goals", "unlabeled_goals", "self_goals")
    }


def visualize_reachability(
    agent,
    dataset,
    epoch,
    save_dir,
    env_name,
    num_viz_points=5000,
    num_anchors=9,
):
    """Visualize reachability landscapes from multiple anchor states.
    
    For maze tasks: Use xy coordinates (first 2 dims)
    For other tasks: Use PCA to reduce to 2D
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Determine if this is a maze task
    is_maze = 'maze' in env_name.lower()
    
    # Get all states from dataset
    all_states_list = [traj for traj in dataset.trajectories]
    all_states = np.concatenate(all_states_list, axis=0)
    
    # Sample subset of points for efficiency
    if all_states.shape[0] > num_viz_points:
        sample_indices = np.random.choice(all_states.shape[0], size=num_viz_points, replace=False)
        sampled_states = all_states[sample_indices]
    else:
        sampled_states = all_states
    
    # Extract goals from sampled states (fallback to identity if no phi provided)
    if hasattr(dataset, 'phi'):
        sampled_goals = dataset.phi(sampled_states)
    else:
        sampled_goals = sampled_states
    
    # Determine visualization coordinates
    if is_maze:
        # For maze tasks, use xy coordinates (first 2 dimensions)
        viz_coords = sampled_states[:, :2]
        coord_labels = ['X Position', 'Y Position']
        print(f"Visualizing maze task using XY coordinates")
    else:
        # For non-maze tasks, use PCA
        print(f"Applying PCA to reduce {sampled_states.shape[1]}D states to 2D")
        pca = PCA(n_components=2)
        viz_coords = pca.fit_transform(sampled_states)
        explained_var = pca.explained_variance_ratio_
        coord_labels = [
            f'PC1 ({explained_var[0]:.1%} var)',
            f'PC2 ({explained_var[1]:.1%} var)'
        ]
        print(f"PCA explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var.sum():.2%}")
    
    # Setup subplots
    n_rows = int(np.ceil(np.sqrt(num_anchors)))
    n_cols = int(np.ceil(num_anchors / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if num_anchors == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Select anchor states (uniformly from trajectory starts)
    start_states = np.stack([traj[0] for traj in dataset.trajectories], axis=0)
    if start_states.shape[0] < num_anchors:
        anchor_indices = np.random.choice(start_states.shape[0], size=num_anchors, replace=True)
    else:
        anchor_indices = np.linspace(0, start_states.shape[0] - 1, num=num_anchors, dtype=int)
    
    # If using PCA, transform anchor states too
    if not is_maze:
        anchor_states_viz = pca.transform(start_states[anchor_indices])
    
    for plot_idx in range(num_anchors):
        ax = axes[plot_idx]
        
        anchor_state = start_states[anchor_indices[plot_idx]]
        
        # Compute reachability for all sampled goals from this anchor
        anchor_batch = np.tile(anchor_state[None, :], (sampled_goals.shape[0], 1))
        
        # Batch evaluation (single device call)
        reachability_scores = agent.predict_reachability(anchor_batch, sampled_goals)
        reachability_scores = np.asarray(reachability_scores).reshape(-1)
        
        # Plot reachability landscape
        scatter = ax.scatter(
            viz_coords[:, 0],
            viz_coords[:, 1],
            c=reachability_scores,
            cmap='RdYlGn',
            s=10,
            alpha=0.7,
            vmin=0.0,
            vmax=1.0,
        )
        
        # Mark anchor state
        if is_maze:
            anchor_coord = anchor_state[:2]
        else:
            anchor_coord = anchor_states_viz[plot_idx]
        
        ax.scatter(
            anchor_coord[0],
            anchor_coord[1],
            marker='*',
            s=500,
            color='blue',
            edgecolors='white',
            linewidths=2.5,
            label='Anchor',
            zorder=10,
        )
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Reachability', fontsize=11)
        
        ax.set_xlabel(coord_labels[0], fontsize=11)
        ax.set_ylabel(coord_labels[1], fontsize=11)
        ax.set_title(f'Anchor #{plot_idx + 1}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for plot_idx in range(num_anchors, len(axes)):
        axes[plot_idx].axis('off')
    
    title = f'Reachability Landscapes @ Step {epoch}'
    if not is_maze:
        title += ' (PCA projection)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    # Save figure
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f'reachability_step_{epoch:07d}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def main(_):
    # Setup logger
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench-RWS', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    config = FLAGS.agent
    dataset_type = (FLAGS.dataset_type or '').lower()
    train_dataset = None
    val_dataset = None
    viz_env_name = FLAGS.env_name

    if dataset_type == 'ogbench':
        dataset_name = FLAGS.dataset_name or FLAGS.env_name
        if not dataset_name:
            raise ValueError("--dataset_name or --env_name must be provided for dataset_type='ogbench'.")

        print(f"Loading OGBench dataset: {dataset_name} ({FLAGS.dataset_split})")
        trajectories = load_ogbench_trajectories(
            dataset_name,
            split=FLAGS.dataset_split,
            compact_dataset=FLAGS.compact_ogbench,
        )
        train_dataset = ReachabilityGCDataset(
            trajectories=trajectories,
            config=config,
        )
        viz_env_name = dataset_name

        # Try to build a validation dataset if a validation split exists.
        if FLAGS.dataset_split.lower() != 'val':
            try:
                val_trajs = load_ogbench_trajectories(
                    dataset_name,
                    split='val',
                    compact_dataset=FLAGS.compact_ogbench,
                )
            except ValueError:
                val_trajs = None
            if val_trajs:
                val_dataset = ReachabilityGCDataset(
                    trajectories=val_trajs,
                    config=config,
                )
    elif dataset_type == 'maze':
        if not FLAGS.maze_buffer:
            raise ValueError("--maze_buffer must be provided for dataset_type='maze'.")
        buffer_path = Path(FLAGS.maze_buffer)
        if not buffer_path.exists():
            raise FileNotFoundError(f"Maze buffer not found at {buffer_path}.")
        print(f"Loading maze buffer from {buffer_path}")
        trajectories = load_maze_trajectories(buffer_path)
        train_dataset = ReachabilityGCDataset(
            trajectories=trajectories,
            config=config,
        )
        if not viz_env_name:
            viz_env_name = buffer_path.stem
    else:
        raise ValueError(f"Unsupported dataset_type '{FLAGS.dataset_type}'. Expected 'ogbench' or 'maze'.")
    
    # Initialize agent
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    example_reachability = train_dataset.sample_batch(
        batch_size=1,
        num_goals_per_state=FLAGS.num_goals_per_state,
        max_skip_horizon=FLAGS.max_skip_horizon,
        num_skip_states=FLAGS.num_skip_states,
    )['reachability']
    example_observations = example_reachability['states']
    example_goals = example_reachability['positive_goals']
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_observations,
        example_goals,
        config,
    )
    
    # Restore agent if specified
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    
    # Training loop
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    viz_dir = os.path.join(FLAGS.save_dir, 'visualizations')
    
    first_time = time.time()
    last_time = time.time()
    
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent
        batch_dict = train_dataset.sample_batch(
            config['batch_size'], 
            num_goals_per_state=FLAGS.num_goals_per_state,
            max_skip_horizon=FLAGS.max_skip_horizon,
            num_skip_states=FLAGS.num_skip_states,
        )

        # Extract reachability batch
        batch_np = batch_dict['reachability']

        # Convert to device arrays
        batch = _prepare_reachability_batch(batch_np)
        
        agent, update_info = agent.update(batch)
        
        # Log metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            if val_dataset is not None:
                val_batch_dict = val_dataset.sample_batch(
                    config['batch_size'], 
                    num_goals_per_state=FLAGS.num_goals_per_state,
                    max_skip_horizon=FLAGS.max_skip_horizon,
                    num_skip_states=FLAGS.num_skip_states,
                )

                # Extract reachability batch
                val_batch_np = val_batch_dict['reachability']

                # Convert to device arrays
                val_batch = _prepare_reachability_batch(val_batch_np)
                
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            
            train_metrics['time/step_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
        
        # Visualize reachability
        if i == 1 or i % FLAGS.viz_interval == 0:
            visualize_reachability(
                agent=agent,
                dataset=train_dataset,
                epoch=i,
                save_dir=viz_dir,
                env_name=viz_env_name,
                num_viz_points=FLAGS.num_viz_points,
                num_anchors=FLAGS.num_viz_anchors,
            )
        
        # Save checkpoint
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
    
    train_logger.close()


if __name__ == '__main__':
    app.run(main)
