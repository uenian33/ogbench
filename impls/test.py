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
from utils.datasets import Dataset, ReachabilityGCDataset, load_maze_trajectories
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

# Import RWS agent
from agents.rws import RWSAgent

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_type', 'ogbench', 'Dataset type: "ogbench" or "maze".')
flags.DEFINE_string('maze_buffer', 'env/A_star_buffer.pkl', 'Path to maze buffer (if dataset_type=maze).')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# Visualization parameters (replacing eval parameters)
flags.DEFINE_integer('viz_anchors', 9, 'Number of anchor states for visualization.')
flags.DEFINE_integer('viz_samples', 5000, 'Number of samples for visualization.')
flags.DEFINE_integer('viz_dim0', 0, 'First dimension to visualize.')
flags.DEFINE_integer('viz_dim1', 1, 'Second dimension to visualize.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/rws.py', lock_config=False)


def visualize_reachability(
    agent,
    dataset,
    step,
    save_dir,
    plot_dims,
    num_anchors=9,
    num_viz_samples=5000,
):
    """Visualize reachability landscapes from multiple anchor states.
    
    Args:
        agent: RWS agent with trained reachability network.
        dataset: ReachabilityGCDataset.
        step: Current training step.
        save_dir: Directory to save visualization.
        plot_dims: Which dimensions to plot (2D).
        num_anchors: Number of anchor states to visualize.
        num_viz_samples: Maximum number of data points to visualize.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dims = [int(d) for d in plot_dims]
    if len(dims) != 2:
        raise ValueError("plot_dims must contain exactly two indices for 2D visualization.")

    n_rows = int(np.ceil(np.sqrt(num_anchors)))
    n_cols = int(np.ceil(num_anchors / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if num_anchors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    bounds_min, bounds_max = dataset.state_bounds
    start_states = dataset.start_states
    
    # Select anchor indices
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

    for plot_idx in range(num_anchors):
        ax = axes[plot_idx]
        
        anchor_state = start_states[anchor_indices[plot_idx]]
        
        # Convert to JAX arrays
        anchor_state_j = jnp.array(anchor_state)[None, :]  # [1, state_dim]
        all_goals_j = jnp.array(all_goals)  # [N, goal_dim]
        
        # Batch evaluation for efficiency
        batch_size = 1024
        reachability_scores = []
        for i in range(0, all_goals_j.shape[0], batch_size):
            batch_goals = all_goals_j[i:i + batch_size]
            anchor_batch = jnp.tile(anchor_state_j, (batch_goals.shape[0], 1))
            scores = agent.predict_reachability(anchor_batch, batch_goals)
            reachability_scores.append(np.array(scores).reshape(-1))
        
        reachability_scores = np.concatenate(reachability_scores, axis=0)
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


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    
    if FLAGS.dataset_type == 'ogbench':
        # Load OGBench dataset
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

        # Create ReachabilityGCDataset
        train_dataset = ReachabilityGCDataset(
            Dataset.create(**train_dataset), 
            config,
            pi_data_type='gc'
        )
        if val_dataset is not None:
            val_dataset = ReachabilityGCDataset(
                Dataset.create(**val_dataset), 
                config,
                pi_data_type='gc'
            )
    elif FLAGS.dataset_type == 'maze':
        # Load maze dataset
        buffer_path = Path(FLAGS.maze_buffer)
        print(f"Loading maze buffer from {buffer_path}")
        trajectories = load_maze_trajectories(buffer_path)
        
        # Create ReachabilityGCDataset from trajectories
        train_dataset = ReachabilityGCDataset(
            trajectories=trajectories,
            pi_data_type='gc'
        )
        val_dataset = None
        env = None  # No environment for maze dataset
    else:
        raise ValueError(f"Invalid dataset_type: {FLAGS.dataset_type}. Must be 'ogbench' or 'maze'.")

    print(f"Dataset loaded: {len(train_dataset)} transitions")
    print(f"State dimension: {train_dataset.state_dim}")
    print(f"Number of trajectories: {train_dataset.num_trajectories}")

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Sample batch for initialization
    example_full_batch = train_dataset.sample_batch(
        batch_size=1,
        num_goals_per_state=config['num_goals_per_state'],
        max_skip_horizon=config['max_skip_horizon'],
        num_skip_states=config['num_skip_states'],
    )
    example_batch = example_full_batch['reachability']
    
    # Create example observations and dummy actions
    ex_observations = example_batch['states']
    ex_actions = np.zeros((1, 1))  # Dummy actions for compatibility

    agent = RWSAgent.create(
        FLAGS.seed,
        ex_observations,
        ex_actions,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    viz_dir = Path(FLAGS.save_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        full_batch = train_dataset.sample_batch(
            batch_size=config['batch_size'],
            num_goals_per_state=config['num_goals_per_state'],
            max_skip_horizon=config['max_skip_horizon'],
            num_skip_states=config['num_skip_states'],
        )
        batch = full_batch['reachability']
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_full_batch = val_dataset.sample_batch(
                    batch_size=config['batch_size'],
                    num_goals_per_state=config['num_goals_per_state'],
                    max_skip_horizon=config['max_skip_horizon'],
                    num_skip_states=config['num_skip_states'],
                )
                val_batch = val_full_batch['reachability']
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Visualize reachability (replacing evaluation).
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            
            eval_metrics = {}
            
            try:
                # Generate reachability visualization
                visualize_reachability(
                    agent=eval_agent,
                    dataset=train_dataset,
                    step=i,
                    save_dir=viz_dir,
                    plot_dims=[FLAGS.viz_dim0, FLAGS.viz_dim1],
                    num_anchors=FLAGS.viz_anchors,
                    num_viz_samples=FLAGS.viz_samples,
                )
                
                # Log visualization path
                eval_metrics['visualization/path'] = str(viz_dir / f"reachability_step_{i:07d}.png")
                
                # Compute some statistics on reachability predictions
                # Sample a batch and compute mean predictions
                stats_batch = train_dataset.sample_batch(
                    batch_size=min(1000, len(train_dataset)),
                    num_goals_per_state=config['num_goals_per_state'],
                    max_skip_horizon=config['max_skip_horizon'],
                    num_skip_states=config['num_skip_states'],
                )['reachability']
                
                # Predict reachability for positive goals
                pos_reach = eval_agent.predict_reachability(
                    stats_batch['states'], 
                    stats_batch['positive_goals']
                )
                # Predict reachability for unlabeled goals (flattened)
                B, K = stats_batch['unlabeled_goals'].shape[:2]
                unl_reach = eval_agent.predict_reachability(
                    jnp.tile(stats_batch['states'][:, None, :], (1, K, 1)).reshape(B * K, -1),
                    stats_batch['unlabeled_goals'].reshape(B * K, -1)
                ).reshape(B, K)
                
                eval_metrics['evaluation/pos_reach_mean'] = float(jnp.mean(pos_reach))
                eval_metrics['evaluation/pos_reach_std'] = float(jnp.std(pos_reach))
                eval_metrics['evaluation/unl_reach_mean'] = float(jnp.mean(unl_reach))
                eval_metrics['evaluation/unl_reach_std'] = float(jnp.std(unl_reach))
                eval_metrics['evaluation/reach_gap'] = float(jnp.mean(pos_reach) - jnp.mean(unl_reach))
                
            except Exception as e:
                print(f"Visualization failed: {e}")
                eval_metrics['visualization/error'] = str(e)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)


'''
# Train reachability on AntMaze
python test.py \
    --env_name antmaze-giant-stitch-v0 \
    --run_group RWS_AntMaze \
    --seed 42 \
    --train_steps 1000000 \
    --log_interval 5000 \
    --eval_interval 10000 \
    --save_interval 100000 \
    --viz_anchors 9 \
    --viz_samples 5000 \
    --viz_dim0 0 \
    --viz_dim1 1 \
    --agent.batch_size 128 \
    --agent.lr 3e-4 \
    --agent.tau 0.995 \
    --agent.rank_margin -0.0 \
    --agent.lambda_cons 1.0 \
    --agent.num_goals_per_state 10 \
    --agent.num_skip_states 1


python test.py \
    --dataset_type maze \
    --maze_buffer env/A_star_buffer.pkl \
    --run_group RWS_Maze \
    --seed 42 \
    --train_steps 500000 \
    --log_interval 1000 \
    --eval_interval 10000 \
    --save_interval 50000 \
    --viz_dim0 0 \
    --viz_dim1 1 \
    --viz_anchors 9 \
    --viz_samples 5000 \
    --agent.batch_size 128 \
    --agent.lr 3e-4 \
    --agent.tau 0.995 \
    --agent.num_skip_states 10 \
    --agent.num_goals_per_state 1
'''