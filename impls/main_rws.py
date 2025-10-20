"""
Reachability estimator training script - Using RWSAgent
Simplified version that uses the agent infrastructure from rws.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

# Import agent and dataset
from agents.rws import RWSAgent
from utils.datasets import ReachabilityGCDataset, load_maze_trajectories, load_ogbench_trajectories
from utils.flax_utils import save_agent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_hidden_dims(values: Sequence[int]) -> List[int]:
    dims = [int(v) for v in values]
    if any(d <= 0 for d in dims):
        raise ValueError("Hidden layer sizes must be positive integers.")
    return dims


def visualize_reachability(
    agent: RWSAgent,
    dataset: ReachabilityGCDataset,
    epoch: int,
    save_dir: Path,
    plot_dims: Sequence[int],
    num_anchors: int = 4,
    num_viz_samples: int = 5000,
) -> None:
    """
    Visualize reachability landscapes from multiple anchor states.
    
    Args:
        agent: RWSAgent with trained reachability network
        dataset: Reachability dataset
        epoch: Current epoch number
        save_dir: Directory to save visualization
        plot_dims: Which dimensions to plot (2D)
        num_anchors: Number of anchor states to visualize
        num_viz_samples: Maximum number of data points to visualize
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dims = list(plot_dims)
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
    
    if start_states.shape[0] < num_anchors:
        anchor_indices = np.random.choice(start_states.shape[0], size=num_anchors, replace=True)
    else:
        anchor_indices = np.linspace(0, start_states.shape[0] - 1, num=num_anchors, dtype=int)

    # Get all states and goals, then sample if needed
    _, all_goals, all_states = dataset.get_anchor_state_and_all_goals(anchor_idx=None)
    
    # Sample a subset if dataset is too large
    total_points = all_goals.shape[0]
    if total_points > num_viz_samples:
        print(f"Sampling {num_viz_samples} out of {total_points} points for visualization")
        sample_indices = np.random.choice(total_points, size=num_viz_samples, replace=False)
        all_goals = all_goals[sample_indices]
        all_states = all_states[sample_indices]
    else:
        print(f"Visualizing all {total_points} points")

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
    fig.suptitle(f"Reachability Landscapes @ Epoch {epoch} {viz_info}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"reachability_epoch_{epoch:04d}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_reachability(
    agent: RWSAgent,
    dataset: ReachabilityGCDataset,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    num_goals_per_state: int,
    num_skip_states: int,
    max_skip_horizon: Optional[int],
    viz_every: int,
    viz_dims: Sequence[int],
    viz_dir: Path,
    viz_samples: int,
    save_every: int,
    save_dir: Path,
) -> RWSAgent:
    """Core training loop for the reachability estimator."""
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    total_transitions = len(dataset)
    skip_info = f", max skip h={max_skip_horizon}" if max_skip_horizon else " (1-step only)"
    print(f"Dataset transitions: {total_transitions}")
    print(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch, batch size {batch_size}{skip_info}.")

    for epoch in tqdm(range(1, epochs + 1)):
        metrics_acc = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), leave=False):
            # Sample batch from dataset
            batch_dict = dataset.sample_batch(
                batch_size, 
                num_goals_per_state=num_goals_per_state,
                max_skip_horizon=max_skip_horizon,
                num_skip_states=num_skip_states,
            )

            # Extract reachability batch
            batch = batch_dict['reachability']
            
            # Training step using agent
            agent, metrics = agent.update(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                metrics_acc[key] += float(value)

        # Average metrics
        for key in metrics_acc:
            metrics_acc[key] /= steps_per_epoch

        # Print metrics with correct keys (rws/ prefix)
        print(
            f"Epoch {epoch:04d} | "
            f"loss {metrics_acc.get('rws/loss_total', 0.0):.4f} | "
            f"rank {metrics_acc.get('rws/loss_rank', 0.0):.4f} | "
            f"cons {metrics_acc.get('rws/loss_cons', 0.0):.4f} | "
            f"pos {metrics_acc.get('rws/pred_pos', 0.0):.3f} | "
            f"unl {metrics_acc.get('rws/pred_unl', 0.0):.3f}"
        )

        if viz_every > 0 and epoch % viz_every == 0:
            visualize_reachability(
                agent,
                dataset,
                epoch=epoch,
                save_dir=viz_dir,
                plot_dims=viz_dims,
                num_anchors=9,
                num_viz_samples=viz_samples,
            )

        if save_every > 0 and epoch % save_every == 0:
            save_agent(agent, save_dir, epoch)
    
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reachability estimator using RWSAgent.")

    # Dataset arguments
    parser.add_argument("--dataset-type", choices=["ogbench", "maze"], required=True, help="Source dataset.")
    parser.add_argument("--dataset-name", type=str, help="OGBench dataset name.")
    parser.add_argument("--dataset-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--compact-ogbench", action="store_true")
    parser.add_argument("--maze-buffer", type=str, default="env/A_star_buffer.pkl")

    # Network architecture
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    
    # Loss hyperparameters
    parser.add_argument("--rank-margin", type=float, default=-0.05)
    parser.add_argument("--lambda-cons", type=float, default=1.0)
    
    # Dataset sampling
    parser.add_argument("--num-goals-per-state", type=int, default=4)
    parser.add_argument("--max-skip-horizon", type=int, default=None)
    parser.add_argument("--num-skip-states", type=int, default=3)

    # Visualization
    parser.add_argument("--viz-every", type=int, default=1)
    parser.add_argument("--viz-dims", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--viz-dir", type=str, default="reachability_viz")
    parser.add_argument("--viz-samples", type=int, default=5000)

    # Saving
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="reachability_checkpoints")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Load dataset
    if args.dataset_type == "ogbench":
        if not args.dataset_name:
            raise ValueError("--dataset-name must be provided for dataset-type 'ogbench'.")
        print(f"Loading OGBench dataset: {args.dataset_name} ({args.dataset_split})")
        trajectories = load_ogbench_trajectories(
            args.dataset_name,
            split=args.dataset_split,
            compact_dataset=args.compact_ogbench,
        )
    else:
        buffer_path = Path(args.maze_buffer)
        print(f"Loading maze buffer from {buffer_path}")
        trajectories = load_maze_trajectories(buffer_path)

    # Create ReachabilityGCDataset
    dataset = ReachabilityGCDataset(
        trajectories=trajectories,
    )

    print(f"Dataset created with {dataset.num_trajectories} trajectories")
    print(f"State dimension: {dataset.state_dim}")
    print(f"Total transitions: {len(dataset)}")

    # Create agent configuration
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    
    config = {
        'agent_name': 'rws',
        'lr': args.lr,
        'batch_size': args.batch_size,
        'value_hidden_dims': tuple(hidden_dims),
        'layer_norm': True,
        'tau': args.tau,
        'rank_margin': args.rank_margin,
        'lambda_cons': args.lambda_cons,
        'num_goals_per_state': args.num_goals_per_state,
        'num_skip_states': args.num_skip_states,
        'max_skip_horizon': args.max_skip_horizon,
        'dataset_class': 'ReachabilityGCDataset',
        'discount': 0.99,
        'value_p_curgoal': 0.0,
        'value_p_trajgoal': 1.0,
        'value_p_randomgoal': 0.0,
        'value_geom_sample': False,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 0.5,
        'actor_p_randomgoal': 0.5,
        'actor_geom_sample': False,
        'gc_negative': False,
        'p_aug': None,
        'frame_stack': None,
        'encoder': None,
        'discrete': False,  # IMPORTANT: Must be False for continuous state spaces
    }

    # Create example batch for agent initialization
    example_batch = dataset.sample_batch(
        batch_size=1,
        num_goals_per_state=args.num_goals_per_state,
        max_skip_horizon=args.max_skip_horizon,
        num_skip_states=args.num_skip_states,
    )['reachability']
    
    ex_observations = example_batch['states']
    ex_actions = np.zeros((1, dataset.state_dim))  # Dummy actions with proper shape

    # Create agent
    agent = RWSAgent.create(
        args.seed,
        ex_observations,
        ex_actions,
        config,
    )

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(len(dataset) // args.batch_size, 1)

    viz_dir = Path(args.viz_dir)
    save_dir = Path(args.save_dir)

    print("\nTraining configuration:")
    print(json.dumps({
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "rank_margin": args.rank_margin,
        "lambda_cons": args.lambda_cons,
        "num_goals_per_state": args.num_goals_per_state,
        "max_skip_horizon": args.max_skip_horizon,
        "num_skip_states": args.num_skip_states,
        "viz_samples": args.viz_samples,
    }, indent=2))

    # Train
    agent = train_reachability(
        agent,
        dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        num_goals_per_state=args.num_goals_per_state,
        max_skip_horizon=args.max_skip_horizon,
        num_skip_states=args.num_skip_states,
        viz_every=args.viz_every,
        viz_dims=args.viz_dims,
        viz_dir=viz_dir,
        viz_samples=args.viz_samples,
        save_every=args.save_every,
        save_dir=save_dir,
    )

    print(f"\nTraining complete! Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()


"""
Example usage:

# Train on maze dataset
python main_rws.py \
    --dataset-type maze \
    --maze-buffer env/A_star_buffer.pkl \
    --hidden-dims 256 256 256 \
    --epochs 500 \
    --batch-size 128 \
    --num-skip-states 50 \
    --num-goals-per-state 4

# Train on OGBench dataset
python main_rws.py \
    --dataset-type ogbench \
    --dataset-name antmaze-large-navigate-v0 \
    --hidden-dims 256 256 256 \
    --epochs 500 \
    --batch-size 128 \
    --num-skip-states 50 \
    --num-goals-per-state 4
"""