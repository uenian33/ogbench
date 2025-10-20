"""
Reachability estimator training script - JAX/FLAX VERSION with GCBilinearValue
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from flax.training import train_state
from tqdm import tqdm


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


def dataset_dict_to_trajectories(dataset: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Convert dataset dictionary (regular or compact) to trajectory list."""
    if "observations" not in dataset:
        raise KeyError("Dataset dictionary must contain 'observations'.")

    obs = np.asarray(dataset["observations"], dtype=np.float32)
    n_steps = obs.shape[0]

    terminals = dataset.get("terminals") or dataset.get("dones")
    if terminals is None:
        terminals = np.zeros(n_steps, dtype=bool)
    else:
        terminals = np.asarray(terminals, dtype=bool)

    timeouts = dataset.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros(n_steps, dtype=bool)
    else:
        timeouts = np.asarray(timeouts, dtype=bool)

    if "next_observations" in dataset:
        next_obs = np.asarray(dataset["next_observations"], dtype=np.float32)
        if next_obs.shape[0] != n_steps:
            raise ValueError("observations and next_observations must have the same length.")

        trajectories: List[np.ndarray] = []
        episode_start = 0

        for idx in range(n_steps):
            done = bool(terminals[idx]) or bool(timeouts[idx])
            if done:
                states = obs[episode_start : idx + 1]
                final_state = next_obs[idx : idx + 1]
                traj = np.concatenate([states, final_state], axis=0)
                if traj.shape[0] >= 2:
                    trajectories.append(traj.astype(np.float32))
                episode_start = idx + 1

        if episode_start < n_steps:
            states = obs[episode_start:]
            final_state = next_obs[-1:]
            traj = np.concatenate([states, final_state], axis=0)
            if traj.shape[0] >= 2:
                trajectories.append(traj.astype(np.float32))

        if not trajectories:
            raise ValueError("No trajectories could be reconstructed from dataset.")

        return trajectories

    valids = dataset.get("valids")
    if valids is None:
        raise KeyError("Compact dataset must include 'valids' when 'next_observations' is absent.")
    valids = np.asarray(valids, dtype=bool)
    if valids.shape[0] != n_steps:
        raise ValueError("'valids' must have the same length as 'observations'.")

    trajectories: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for idx in range(n_steps):
        current.append(obs[idx])
        done = (not bool(valids[idx])) or bool(terminals[idx]) or bool(timeouts[idx])
        if done:
            if len(current) >= 2:
                trajectories.append(np.stack(current, axis=0).astype(np.float32))
            current = []

    if current and len(current) >= 2:
        trajectories.append(np.stack(current, axis=0).astype(np.float32))

    if not trajectories:
        raise ValueError("No trajectories could be reconstructed from compact dataset.")

    return trajectories


def load_ogbench_trajectories(
    dataset_name: str,
    split: str = "train",
    compact_dataset: bool = False,
) -> List[np.ndarray]:
    try:
        import ogbench
    except ImportError as exc:
        raise ImportError(
            "ogbench is required for --dataset-type ogbench. Install with `pip install ogbench`."
        ) from exc

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name,
        compact_dataset=compact_dataset,
    )
    _ = env
    dataset = train_dataset if split == "train" else val_dataset
    if dataset is None:
        raise ValueError(f"Unable to load OGBench dataset split '{split}'.")
    return dataset_dict_to_trajectories(dataset)


def load_maze_trajectories(buffer_path: Path) -> List[np.ndarray]:
    import pickle

    with open(buffer_path, "rb") as handle:
        data = pickle.load(handle)

    if "o" not in data:
        raise KeyError(f"Expected key 'o' in maze buffer at {buffer_path}.")

    obs = np.asarray(data["o"], dtype=np.float32)
    if obs.ndim != 3:
        raise ValueError(f"'o' tensor must have shape (num_traj, horizon, obs_dim), got {obs.shape}.")

    trajectories = [traj for traj in obs if traj.shape[0] >= 2]
    if not trajectories:
        raise ValueError("Maze buffer does not contain any valid trajectories.")
    return trajectories


class OfflineReachabilityDataset:
    """Lightweight container for offline trajectories with sampling utilities."""

    def __init__(
        self,
        trajectories: Sequence[np.ndarray],
        goal_dim: int,
        epsilon: Optional[float] = None,
        epsilon_multiplier: float = 2.0,
        goal_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if not trajectories:
            raise ValueError("At least one trajectory is required.")

        traj_list: List[np.ndarray] = []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim != 2:
                raise ValueError(f"Trajectory must be 2-D, got shape {traj.shape}.")
            if traj.shape[0] < 2:
                continue
            traj_list.append(traj)

        if not traj_list:
            raise ValueError("No trajectories with at least two states were provided.")

        self.trajectories = traj_list
        self.num_trajectories = len(traj_list)
        self.state_dim = traj_list[0].shape[1]
        self.goal_dim = goal_dim

        if goal_fn is None:
            self._goal_fn: Callable[[np.ndarray], np.ndarray] = lambda arr: arr[..., : self.goal_dim]
        else:
            self._goal_fn = goal_fn

        self._states: List[np.ndarray] = []
        self._next_states: List[np.ndarray] = []
        self._traj_ids: List[int] = []
        self._step_ids: List[int] = []
        self._indices_by_traj: List[List[int]] = [[] for _ in range(self.num_trajectories)]
        self._build_transition_index()

        self._states_np = np.asarray(self._states, dtype=np.float32)
        self._next_states_np = np.asarray(self._next_states, dtype=np.float32)
        self._traj_ids_np = np.asarray(self._traj_ids, dtype=np.int64)
        self._step_ids_np = np.asarray(self._step_ids, dtype=np.int64)
        
        # Compute single-step distance in GOAL SPACE
        goal_curr = self.phi(self._states_np)
        goal_next = self.phi(self._next_states_np)
        single_step_distances = np.linalg.norm(goal_next - goal_curr, axis=1)
        
        self.max_single_step_distance = float(single_step_distances.max())
        self.mean_single_step_distance = float(single_step_distances.mean())
        self.median_single_step_distance = float(np.median(single_step_distances))
        self.p95_single_step_distance = float(np.percentile(single_step_distances, 95))
        
        if epsilon is None:
            self.epsilon = self.p95_single_step_distance * epsilon_multiplier
            print(f"Auto-computed epsilon: {self.epsilon:.6f}")
            print(f"  max_step_distance:    {self.max_single_step_distance:.6f}")
            print(f"  p95_step_distance:    {self.p95_single_step_distance:.6f}")
            print(f"  mean_step_distance:   {self.mean_single_step_distance:.6f}")
            print(f"  median_step_distance: {self.median_single_step_distance:.6f}")
        else:
            self.epsilon = epsilon
            print(f"Using provided epsilon: {self.epsilon:.6f}")
        
        self._goal_pool = np.concatenate([self._goal_fn(traj) for traj in self.trajectories], axis=0)
        self._final_goals = np.stack([self._goal_fn(traj[-1]) for traj in self.trajectories], axis=0)
        self._self_goals = self._goal_fn(self._states_np)
        self._start_states = np.stack([traj[0] for traj in self.trajectories], axis=0).astype(np.float32)

        self._state_min = self._states_np.min(axis=0)
        self._state_max = self._states_np.max(axis=0)

    def _build_transition_index(self) -> None:
        for traj_id, traj in enumerate(self.trajectories):
            steps = traj.shape[0] - 1
            for step in range(steps):
                self._states.append(traj[step])
                self._next_states.append(traj[step + 1])
                self._traj_ids.append(traj_id)
                self._step_ids.append(step)
                self._indices_by_traj[traj_id].append(len(self._states) - 1)

    def __len__(self) -> int:
        return self._states_np.shape[0]

    @property
    def state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._state_min, self._state_max

    def phi(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(self._goal_fn(arr), dtype=np.float32)

    def sample_batch(
        self, 
        batch_size: int,
        num_goals_per_state: int = 4,
        max_skip_horizon: Optional[int] = None,
        num_skip_states: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Sample minibatch with multi-step skip consistency.
        
        CORRECTED SAMPLING:
        1. Sample batch_size random transitions
        2. For each transition, sample 1 positive goal from a random future state
        3. For each transition, sample num_skip_states skip states with horizons 
           h ∈ [1, positive_goal_idx - current_step] (between current and goal)
        4. Sample num_goals_per_state unlabeled goals from global pool
        """
        idx = np.random.randint(0, len(self), size=batch_size)
        states = self._states_np[idx]
        traj_ids = self._traj_ids_np[idx]
        step_ids = self._step_ids_np[idx]

        # Initialize arrays
        skip_states = np.empty((batch_size, num_skip_states, self.state_dim), dtype=np.float32)
        pos_goals = np.empty((batch_size, self.goal_dim), dtype=np.float32)
        pos_goal_indices = np.empty(batch_size, dtype=np.int64)

        # Sample positive goals FIRST, then skip states
        for i in range(batch_size):
            traj_id = traj_ids[i]
            step = step_ids[i]
            traj = self.trajectories[traj_id]
            traj_length = traj.shape[0]
            
            # Sample positive goal from future states
            # future_step ∈ [step + 1, traj_length)
            future_step = np.random.randint(step + 1, traj_length)
            pos_goals[i] = self._goal_fn(traj[future_step])
            pos_goal_indices[i] = future_step
            
            # Sample skip states between current step and positive goal
            # h ∈ [1, future_step - step]
            max_h = future_step - step
            
            if max_h <= 0:
                # Edge case: use next state
                skip_states[i, :] = self._next_states_np[idx[i]]
            else:
                # Apply optional max_skip_horizon constraint
                if max_skip_horizon is not None:
                    max_h = min(max_h, max_skip_horizon)
                
                # Sample num_skip_states random horizons
                for m in range(num_skip_states):
                    if max_h == 0:
                        h = 0
                    else:
                        h = np.random.randint(1, max_h + 1)
                    skip_states[i, m] = traj[step + h]

        # Sample unlabeled goals from global pool
        unl_goals = np.empty((batch_size, num_goals_per_state, self.goal_dim), dtype=np.float32)
        for i in range(batch_size):
            rand_indices = np.random.randint(0, self._goal_pool.shape[0], size=num_goals_per_state)
            unl_goals[i] = self._goal_pool[rand_indices]

        self_goals = self._self_goals[idx]

        return {
            "states": states,
            "skip_states": skip_states,
            "positive_goals": pos_goals,
            "unlabeled_goals": unl_goals,
            "self_goals": self_goals,
            "traj_ids": traj_ids,
        }

    def get_anchor_state_and_all_goals(
        self, 
        anchor_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get one anchor state and all goal positions for visualization."""
        if anchor_idx is None:
            anchor_idx = np.random.randint(0, self.num_trajectories)
            anchor_state = self.trajectories[anchor_idx][0]
        else:
            anchor_state = self._states_np[anchor_idx]
        
        all_goals = self._goal_pool
        all_states = np.concatenate([traj for traj in self.trajectories], axis=0)
        
        return anchor_state.astype(np.float32), all_goals.astype(np.float32), all_states.astype(np.float32)

    @property
    def start_states(self) -> np.ndarray:
        return self._start_states


# ============================================================================
# Network Components
# ============================================================================

class MLP(nn.Module):
    """Basic MLP with optional layer normalization."""
    hidden_dims: Sequence[int]
    activate_final: bool = True
    layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim)(x)
            
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)
            elif self.layer_norm and i == len(self.hidden_dims) - 1:
                # Apply layer norm to final layer if not activating
                x = nn.LayerNorm()(x)
                
        return x


def ensemblize(module_class, num_ensembles: int):
    """Create an ensemble of identical modules by vmapping over an extra axis."""
    class EnsembleModule(nn.Module):
        """Ensemble wrapper that vmaps over the first axis."""
        module_args: tuple = ()
        module_kwargs: dict = field(default_factory=dict)
        
        @nn.compact
        def __call__(self, *args, **kwargs):
            # Create num_ensembles copies of the base module
            ensemble_module = nn.vmap(
                module_class,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                out_axes=0,
                axis_size=num_ensembles
            )(*self.module_args, **self.module_kwargs)
            
            return ensemble_module(*args, **kwargs)
    
    def wrapper(*args, **kwargs):
        return EnsembleModule(module_args=args, module_kwargs=kwargs)
    
    return wrapper


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        #if self.value_exp:
        #    v = jnp.exp(v)

        #v = jax.nn.sigmoid(v)

        if info:
            return v, phi, psi
        else:
            return v


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

# ============================================================================
# Training Infrastructure
# ============================================================================

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
    epsilon: float,
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
        pred_pos = state.apply_fn(params, states, pos_goals)  # [2, B] if ensemble, else [B]
        if pred_pos.ndim > 1:
            # If ensemble, take mean over ensemble dimension (axis 0)
            pred_pos = pred_pos.mean(axis=0)  # [B]
        pred_pos = pred_pos[:, None]  # [B, 1]
        
        # Compute predictions for all K unlabeled goals
        states_expanded = jnp.expand_dims(states, 1)  # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1))  # [B, K, state_dim]
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        pred_unl = state.apply_fn(params, states_flat, unl_goals_flat)  # [2, B*K] if ensemble
        if pred_unl.ndim > 1:
            # If ensemble, take mean over ensemble dimension (axis 0)
            pred_unl = pred_unl.mean(axis=0)  # [B*K]
        pred_unl = pred_unl.reshape(B, K)  # [B, K]
        
        # Ranking loss
        rank_logits = pred_pos - pred_unl.mean(axis=1, keepdims=True) - rank_margin
        rank_loss = jnp.mean(jax.nn.softplus(-rank_logits))
        
        # === 2. MULTI-STEP CONSISTENCY LOSS WITH MAX-POOLING ===
        
        # For POSITIVE goals:
        # Compute r̄(s_{t+hi}, g^+) for all M skip states
        skip_states_flat = skip_states.reshape(B * M, -1)  # [B*M, state_dim]
        pos_goals_expanded = jnp.expand_dims(pos_goals, 1)  # [B, 1, goal_dim]
        pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1))  # [B, M, goal_dim]
        pos_goals_flat = pos_goals_expanded.reshape(B * M, -1)  # [B*M, goal_dim]
        
        # Use target network (no gradients)
        target_pos_all = state.apply_fn(state.target_params, skip_states_flat, pos_goals_flat)
        if target_pos_all.ndim > 1:
            # If ensemble, take mean over ensemble dimension (axis 0)
            target_pos_all = target_pos_all.mean(axis=0)  # [B*M]
        target_pos_all = target_pos_all.reshape(B, M)  # [B, M]
        #target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B, 1]
        
        # L2 loss: mean((target_pos_max - pred_pos)^2)
        cons_pos = -jnp.mean(target_pos_all)  - jnp.mean(pred_pos)
        
        # For UNLABELED goals:
        # Compute r̄(s_{t+hi}, g_unl) for all M skip states and K unlabeled goals
        skip_states_expanded = jnp.expand_dims(skip_states, 2)  # [B, M, 1, state_dim]
        skip_states_expanded = jnp.tile(skip_states_expanded, (1, 1, K, 1))  # [B, M, K, state_dim]
        skip_states_expanded_flat = skip_states_expanded.reshape(B * M * K, -1)  # [B*M*K, state_dim]
        
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)  # [B, 1, K, goal_dim]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1))  # [B, M, K, goal_dim]
        unl_goals_expanded_flat = unl_goals_expanded.reshape(B * M * K, -1)  # [B*M*K, goal_dim]
        
        # Use target network
        target_unl_all = state.apply_fn(state.target_params, skip_states_expanded_flat, unl_goals_expanded_flat)
        if target_unl_all.ndim > 1:
            # If ensemble, take mean over ensemble dimension (axis 0)
            target_unl_all = target_unl_all.mean(axis=0)  # [B*M*K]
        target_unl_all = target_unl_all.reshape(B, M, K)  # [B, M, K]
        target_unl_max = jnp.max(target_unl_all, axis=1)  # [B, K] - max over M skip states
        
        # L2 loss: mean((target_unl_max - pred_unl)^2)
        #cons_unl = jnp.mean((jax.lax.stop_gradient(target_unl_max) - pred_unl) ** 2)
        
        cons_unl = jnp.mean(jax.nn.relu((jax.lax.stop_gradient(target_unl_max) - pred_unl)))
        # Total consistency loss
        consistency_loss = cons_pos + cons_unl
        
        # === TOTAL LOSS ===
        total_loss = rank_loss + lambda_cons * consistency_loss
        # Note: lambda_goal and lambda_mass terms are commented out in original
        
        metrics = {
            "loss_total": total_loss,
            "loss_rank": rank_loss,
            "loss_cons": consistency_loss,
            "loss_cons_pos": cons_pos,
            "loss_cons_unl": cons_unl,
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
    dataset: OfflineReachabilityDataset,
    epoch: int,
    save_dir: Path,
    plot_dims: Sequence[int],
    num_anchors: int = 4,
) -> None:
    """Visualize reachability landscapes from multiple anchor states."""
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

    for plot_idx in range(num_anchors):
        ax = axes[plot_idx]
        
        anchor_state = start_states[anchor_indices[plot_idx]]
        _, all_goals, all_states = dataset.get_anchor_state_and_all_goals(anchor_idx=None)
        
        # Convert to JAX arrays
        anchor_state_j = jnp.array(anchor_state)[None, :]  # [1, state_dim]
        all_goals_j = jnp.array(all_goals)  # [N, goal_dim]
        
        # Batch evaluation
        batch_size = 1024
        reachability_scores = []
        for i in range(0, all_goals_j.shape[0], batch_size):
            batch_goals = all_goals_j[i:i + batch_size]
            anchor_batch = jnp.tile(anchor_state_j, (batch_goals.shape[0], 1))
            scores = state.apply_fn(state.params, anchor_batch, batch_goals)
            if scores.ndim > 1:
                # If ensemble, take mean over ensemble dimension (axis 0)
                scores = scores.mean(axis=0)
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
            vmin=reachability_scores.min(),
            vmax=reachability_scores.max(),
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
    
    fig.suptitle(f"Reachability Landscapes @ Epoch {epoch}", fontsize=14, fontweight="bold")
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
    dataset: OfflineReachabilityDataset,
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
        dataset.goal_dim,
    )
    
    total_transitions = len(dataset)
    skip_info = f", max skip h={max_skip_horizon}" if max_skip_horizon else " (dynamic based on goal)"
    print(f"Dataset transitions: {total_transitions}")
    print(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch, batch size {batch_size}{skip_info}.")

    for epoch in tqdm(range(1, epochs + 1)):
        metrics_acc = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), leave=False):
            # Sample batch (numpy)
            batch_np = dataset.sample_batch(
                batch_size, 
                num_goals_per_state=num_goals_per_state,
                max_skip_horizon=max_skip_horizon,
                num_skip_states=num_skip_states,
            )
            
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
                dataset.epsilon,
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
            f"cons {metrics_acc['loss_cons']:.4f} "
            f"(pos {metrics_acc['loss_cons_pos']:.4f} + unl {metrics_acc['loss_cons_unl']:.4f}) | "
            f"pred_pos {metrics_acc['pred_pos']:.3f} | "
            f"pred_unl {metrics_acc['pred_unl']:.3f}"
        )

        if viz_every > 0 and epoch % viz_every == 0:
            visualize_reachability(
                state,
                dataset,
                epoch=epoch,
                save_dir=viz_dir,
                plot_dims=viz_dims,
                num_anchors=9,
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

    parser.add_argument("--goal-dim", type=int, default=2)
    parser.add_argument("--goal-threshold", type=float, default=None)
    parser.add_argument("--epsilon-multiplier", type=float, default=2.0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension for bilinear network")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema-tau", type=float, default=0.995)
    
    parser.add_argument("--rank-margin", type=float, default=-0.15)
    parser.add_argument("--lambda-cons", type=float, default=1.0)
    parser.add_argument("--lambda-goal", type=float, default=0.0)
    parser.add_argument("--lambda-mass", type=float, default=0.0)
    
    parser.add_argument("--num-goals-per-state", type=int, default=4)
    parser.add_argument("--max-skip-horizon", type=int, default=None)
    parser.add_argument("--num-skip-states", type=int, default=3)

    parser.add_argument("--viz-every", type=int, default=10)
    parser.add_argument("--viz-dims", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--viz-dir", type=str, default="reachability_viz_jax")

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

    dataset = OfflineReachabilityDataset(
        trajectories,
        goal_dim=args.goal_dim,
        epsilon=args.goal_threshold,
        epsilon_multiplier=args.epsilon_multiplier,
    )

    print(f"Using JAX backend")

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = GCBilinearValue(
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        layer_norm=True,
        ensemble=True,
        value_exp=False,
    )

    

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
        "epsilon": dataset.epsilon,
        "latent_dim": args.latent_dim,
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
        save_every=args.save_every,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
