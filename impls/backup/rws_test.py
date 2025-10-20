"""
Reachability estimator training script - JAX/FLAX VERSION
Imports ReachabilityGCDataset from datasets.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from flax.training import train_state
from tqdm import tqdm

# Import from datasets.py
from utils.datasets import ReachabilityGCDataset, load_maze_trajectories, load_ogbench_trajectories, dataset_dict_to_trajectories


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_hidden_dims(values: Sequence[int]) -> List[int]:
    dims = [int(v) for v in values]
    if any(d <= 0 for d in dims):
        raise ValueError("Hidden layer sizes must be positive integers.")
    return dims


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



class ReachabilityNet(nn.Module):
    """Flax MLP reachability estimator with sigmoid output."""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, states: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with automatic shape handling.
        
        Args:
            states: [..., state_dim]
            goals: [..., goal_dim]
        
        Returns:
            reachability: [..., 1]
        """
        x = jnp.concatenate([states, goals], axis=-1)
        
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        logits = nn.Dense(1)(x)
        return nn.sigmoid(logits)


class TrainState(train_state.TrainState):
    """Extended train state with target network parameters."""
    target_params: Any


def create_train_state(
    rng: jnp.ndarray,
    model: nn.Module,
    learning_rate: float,
    state_dim: int,
    goal_dim: int,
) -> TrainState:
    """Create initial training state with optimizer and target network."""
    # Initialize parameters
    dummy_state = jnp.ones((1, state_dim))
    dummy_goal = jnp.ones((1, goal_dim))
    params = model.init(rng, dummy_state, dummy_goal)
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    
    # Create train state with target params (initially same as params)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        target_params=jax.tree_util.tree_map(lambda x: x.copy(), params),
    )


def update_target_network(
    params: Any,
    target_params: Any,
    tau: float,
) -> Any:
    """EMA update of target network parameters."""
    return jax.tree_util.tree_map(
        lambda p, tp: tau * tp + (1.0 - tau) * p,
        params,
        target_params,
    )


@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rank_margin: float,
    lambda_cons: float,
    lambda_goal: float,
    lambda_mass: float,
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step (JIT compiled)."""
    
    def loss_fn(params):
        states = batch["states"]  # [B, state_dim]
        skip_states = batch["skip_states"]  # [B, M, state_dim]
        pos_goals = batch["positive_goals"]  # [B, goal_dim]
        unl_goals = batch["unlabeled_goals"]  # [B, K, goal_dim]
        self_goals = batch["self_goals"]  # [B, goal_dim]
        
        B, M, state_dim = skip_states.shape
        K = unl_goals.shape[1]
        
        # === 1. PU-RANK LOSS ===
        pred_pos = state.apply_fn(params, states, pos_goals)  # [B, 1]
        
        # Compute predictions for all K unlabeled goals
        states_expanded = jnp.expand_dims(states, 1)  # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1))  # [B, K, state_dim]
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        pred_unl = state.apply_fn(params, states_flat, unl_goals_flat).reshape(B, K)
        
        # Ranking loss
        rank_logits = pred_pos - pred_unl.mean(axis=1, keepdims=True) - rank_margin
        rank_loss = jnp.mean(jax.nn.softplus(-rank_logits))
        
        # === 2. MULTI-STEP CONSISTENCY LOSS WITH MAX-POOLING ===
        # For positives: max_i r̄(s_{t+hi}, g^+)
        skip_states_flat = skip_states.reshape(B * M, -1)
        pos_goals_expanded = jnp.expand_dims(pos_goals, 1)  # [B, 1, goal_dim]
        pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1))  # [B, M, goal_dim]
        pos_goals_expanded = pos_goals_expanded.reshape(B * M, -1)
        
        # Use target network (no gradients)
        target_pos_all = state.apply_fn(state.target_params, skip_states_flat, pos_goals_expanded)
        target_pos_all = target_pos_all.reshape(B, M)
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B, 1]
        
        cons_pos = -jnp.mean(target_pos_all) - jnp.mean(pred_pos)

        # For unlabeled: max_i r̄(s_{t+hi}, g_unl)
        skip_states_expanded_unl = jnp.expand_dims(skip_states, 2)  # [B, M, 1, state_dim]
        skip_states_expanded_unl = jnp.tile(skip_states_expanded_unl, (1, 1, K, 1))  # [B, M, K, state_dim]
        skip_states_expanded_unl = skip_states_expanded_unl.reshape(B * M * K, -1)
        
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)  # [B, 1, K, goal_dim]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1))  # [B, M, K, goal_dim]
        unl_goals_expanded = unl_goals_expanded.reshape(B * M * K, -1)
        
        target_unl_all = state.apply_fn(state.target_params, skip_states_expanded_unl, unl_goals_expanded)
        target_unl_all = target_unl_all.reshape(B, M, K)
        target_unl_max = jnp.max(target_unl_all, axis=1)  # [B, K]
        
        cons_unl = jnp.mean(jax.nn.relu((jax.lax.stop_gradient(target_unl_max) - pred_unl)))
        
        consistency_loss = 0.5 * (cons_pos + cons_unl)
        
        # === TOTAL LOSS ===
        total_loss = rank_loss + lambda_cons * consistency_loss
        
        metrics = {
            "loss_total": total_loss,
            "loss_rank": rank_loss,
            "loss_cons": consistency_loss,
            "pred_pos": jnp.mean(pred_pos),
            "pred_unl": jnp.mean(pred_unl),
        }
        
        return total_loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # Gradient clipping
    grads = optax.clip_by_global_norm(1.0).update(grads, state.opt_state, state.params)[0]
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def visualize_reachability(
    state: TrainState,
    dataset: ReachabilityGCDataset,
    epoch: int,
    save_dir: Path,
    plot_dims: Sequence[int],
    num_anchors: int = 4,
    num_viz_samples: int = 5000,
) -> None:
    """
    Visualize reachability landscapes from multiple anchor states.
    Only samples a subset of data points for efficient visualization.
    
    Args:
        state: Training state containing model parameters
        dataset: Reachability dataset
        epoch: Current epoch number
        save_dir: Directory to save visualization
        plot_dims: Which dimensions to plot (2D)
        num_anchors: Number of anchor states to visualize
        num_viz_samples: Maximum number of data points to visualize (default: 5000)
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
            scores = state.apply_fn(state.params, anchor_batch, batch_goals)
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


def save_checkpoint(
    state: TrainState,
    epoch: int,
    save_dir: Path,
) -> None:
    """Save checkpoint to disk."""
    ensure_dir(save_dir)
    ckpt_path = save_dir / f"reachability_epoch_{epoch:04d}.pkl"
    
    import pickle
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "epoch": epoch,
            "params": state.params,
            "target_params": state.target_params,
            "opt_state": state.opt_state,
        }, f)


def train_reachability(
    rng: jnp.ndarray,
    model: nn.Module,
    dataset: ReachabilityGCDataset,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    ema_tau: float,
    rank_margin: float,
    lambda_cons: float,
    lambda_goal: float,
    lambda_mass: float,
    num_goals_per_state: int,
    num_skip_states: int,
    max_skip_horizon: Optional[int],
    viz_every: int,
    viz_dims: Sequence[int],
    viz_dir: Path,
    viz_samples: int,
    save_every: int,
    save_dir: Path,
) -> TrainState:
    """Core training loop for the reachability estimator."""
    
    # Initialize training state
    rng, init_rng = jrandom.split(rng)
    state = create_train_state(
        init_rng,
        model,
        lr,
        dataset.state_dim,
        dataset.state_dim,
    )
    
    total_transitions = len(dataset)
    skip_info = f", max skip h={max_skip_horizon}" if max_skip_horizon else " (1-step only)"
    print(f"Dataset transitions: {total_transitions}")
    print(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch, batch size {batch_size}{skip_info}.")

    for epoch in tqdm(range(1, epochs + 1)):
        metrics_acc = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), leave=False):

            # Sample batch (numpy) - fully vectorized
            batch_dict = dataset.sample_batch(
                batch_size, 
                num_goals_per_state=num_goals_per_state,
                max_skip_horizon=max_skip_horizon,
                num_skip_states=num_skip_states,
            )

            # Extract reachability batch
            batch_np = batch_dict['reachability']

            # Convert to JAX arrays
            batch = {
                k: jnp.array(v) for k, v in batch_np.items()
                if k in ["states", "skip_states", "positive_goals", "unlabeled_goals", "self_goals"]
            }
            
            # Training step
            state, metrics = train_step(
                state,
                batch,
                rank_margin,
                lambda_cons,
                lambda_goal,
                lambda_mass,
            )
            
            # Accumulate metrics
            for key, value in metrics.items():
                metrics_acc[key] += float(value)
            
            # Update target network
            state = state.replace(
                target_params=update_target_network(state.params, state.target_params, ema_tau)
            )

        # Average metrics
        for key in metrics_acc:
            metrics_acc[key] /= steps_per_epoch

        print(
            f"Epoch {epoch:04d} | "
            f"loss {metrics_acc['loss_total']:.4f} | "
            f"rank {metrics_acc['loss_rank']:.4f} | "
            f"cons {metrics_acc['loss_cons']:.4f} | "
            f"pos {metrics_acc['pred_pos']:.3f} | "
            f"unl {metrics_acc['pred_unl']:.3f}"
        )

        if viz_every > 0 and epoch % viz_every == 0:
            visualize_reachability(
                state,
                dataset,
                epoch=epoch,
                save_dir=viz_dir,
                plot_dims=viz_dims,
                num_anchors=9,
                num_viz_samples=viz_samples,
            )

        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(state, epoch, save_dir)
    
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reachability estimator from offline data (JAX version).")

    parser.add_argument("--dataset-type", choices=["ogbench", "maze"], required=True, help="Source dataset.")
    parser.add_argument("--dataset-name", type=str, help="OGBench dataset name.")
    parser.add_argument("--dataset-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--compact-ogbench", action="store_true")
    parser.add_argument("--maze-buffer", type=str, default="env/A_star_buffer.pkl")

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema-tau", type=float, default=0.995)
    
    parser.add_argument("--rank-margin", type=float, default=-0.05)
    parser.add_argument("--lambda-cons", type=float, default=1.0)
    parser.add_argument("--lambda-goal", type=float, default=0.0)
    parser.add_argument("--lambda-mass", type=float, default=0.0)
    
    parser.add_argument("--num-goals-per-state", type=int, default=4)
    parser.add_argument("--max-skip-horizon", type=int, default=None)
    parser.add_argument("--num-skip-states", type=int, default=3)

    parser.add_argument("--viz-every", type=int, default=10)
    parser.add_argument("--viz-dims", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--viz-dir", type=str, default="reachability_viz_jax")
    parser.add_argument("--viz-samples", type=int, default=5000, help="Number of data points to visualize (for large datasets)")

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="reachability_checkpoints_jax")

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

    # Create ReachabilityGCDataset using dataclass initialization
    dataset = ReachabilityGCDataset(
        trajectories=trajectories,
    )

    print(f"Using JAX backend with ReachabilityGCDataset (fully vectorized)")

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = ReachabilityNet(hidden_dims=hidden_dims)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(len(dataset) // args.batch_size, 1)

    viz_dir = Path(args.viz_dir)
    save_dir = Path(args.save_dir)

    print("Training configuration:")
    print(json.dumps({
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ema_tau": args.ema_tau,
        "rank_margin": args.rank_margin,
        "lambda_cons": args.lambda_cons,
        "lambda_goal": args.lambda_goal,
        "lambda_mass": args.lambda_mass,
        "num_goals_per_state": args.num_goals_per_state,
        "max_skip_horizon": args.max_skip_horizon,
        "num_skip_states": args.num_skip_states,
        "viz_samples": args.viz_samples,
    }, indent=2))

    # Create RNG key
    rng = jrandom.PRNGKey(args.seed)

    train_reachability(
        rng,
        model,
        dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_tau=args.ema_tau,
        rank_margin=args.rank_margin,
        lambda_cons=args.lambda_cons,
        lambda_goal=args.lambda_goal,
        lambda_mass=args.lambda_mass,
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


if __name__ == "__main__":
    main()


'''
python rws_test.py     --dataset-type maze     --maze-buffer env/A_star_buffer.pkl     --goal-dim 2     --hidden-dims 256 256 256        --epochs 500     --batch-size 128 --num-skip-states 50


 python rws_test.py \
     --dataset-type ogbench \
     --dataset-name antmaze-medium-navigate-v0 \
     --hidden-dims 256 256 256 \
     --epochs 500 \
     --batch-size 128 \
     --num-skip-states 50

'''
