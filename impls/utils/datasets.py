import dataclasses
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result



def dataset_dict_to_trajectories(dataset: Dict[str, np.ndarray]) -> List[np.ndarray]:
    required_keys = {"observations", "next_observations"}
    missing = required_keys.difference(dataset.keys())
    if missing:
        raise KeyError(f"Dataset dictionary is missing keys: {missing}")

    obs = np.asarray(dataset["observations"], dtype=np.float32)
    next_obs = np.asarray(dataset["next_observations"], dtype=np.float32)
    n_steps = obs.shape[0]

    if next_obs.shape[0] != n_steps:
        raise ValueError("observations and next_observations must have the same length.")

    terminals = dataset.get("terminals")
    if terminals is None:
        terminals = dataset.get("dones")
    if terminals is None:
        terminals = np.zeros(n_steps, dtype=bool)
    else:
        terminals = np.asarray(terminals, dtype=bool)

    timeouts = dataset.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros(n_steps, dtype=bool)
    else:
        timeouts = np.asarray(timeouts, dtype=bool)

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

class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch



@dataclasses.dataclass
class TDInfoNCEDataset(GCDataset):
    """Dataset for TD-InfoNCE agent.
    
    This dataset samples goals following TD-InfoNCE's strategy:
    - future_goal: Future states from the same trajectory (geometrically sampled)
    - intermediate_future_goal: States between current and future_goal
    - random_goal: Random states (implemented as rolled future_goals in a batch)
    
    For critic training:
        - value_goals = random_goals (used as 'g' in Q(s,a,g,s_future))
        - next_observations = immediate next states (positive future states)
        - random_goals (rolled) = negative future states
    
    For actor training:
        - actor_goals = mix of future_goals and random_goals based on config
    """
    
    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch with future_goal, intermediate_future_goal, and random_goal for TD-InfoNCE.
        
        Args:
            batch_size: Batch size.
            idxs: Indices to sample. If None, random indices are sampled.
            evaluation: Whether in evaluation mode (no augmentation).
        
        Returns:
            Batch dictionary with keys:
                - observations, next_observations, actions
                - future_goals: Future states from trajectory (geometric sampling)
                - intermediate_future_goals: States between current and future_goals
                - random_goals: Random goals (rolled future_goals)
                - value_goals: Goals for critic (= random_goals)
                - actor_goals: Goals for actor (mix of future/random)
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        
        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)
        
        # 1. Sample future_goals from trajectory with geometric sampling
        # This follows TD-InfoNCE's future_goal_relabeling
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        
        if self.config['value_geom_sample']:
            # Geometric sampling: P(offset=k) ~ (1-γ) * γ^(k-1)
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)
            future_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling from future
            distances = np.random.rand(batch_size)
            future_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + 
                 final_state_idxs * (1 - distances))
            ).astype(int)
        
        batch['future_goals'] = self.get_observations(future_goal_idxs)
        
        # 2. Sample intermediate_future_goals between current and future_goals
        # Calculate the distance between current state and future goal
        distances = future_goal_idxs - idxs
        intermediate_goal_idxs = np.zeros_like(future_goal_idxs)
        
        for i in range(batch_size):
            if distances[i] > 1:
                # There are intermediate states available
                # Sample uniformly from (idxs[i] + 1) to (future_goal_idxs[i] - 1)
                intermediate_goal_idxs[i] = np.random.randint(
                    idxs[i] + 1,  # exclusive of current state
                    future_goal_idxs[i]  # exclusive of future goal
                )
            else:
                # No intermediate states available (future is next state or same state)
                # Use the future goal itself
                intermediate_goal_idxs[i] = future_goal_idxs[i]
        
        batch['intermediate_future_goals'] = self.get_observations(intermediate_goal_idxs.astype(int))
        
        # 3. Sample random_goals by rolling future_goals
        # This follows TD-InfoNCE's random_goal_relabeling: roll states in batch
        batch['random_goals'] = np.roll(batch['future_goals'], shift=1, axis=0)
        
        # 4. Set value_goals for critic training
        # Critic uses random_goals as the 'g' in Q(s,a,g,s_future)
        batch['value_goals'] = batch['random_goals']
        
        # 5. Set actor_goals based on config
        # Mix future_goals and random_goals based on actor_p_trajgoal and actor_p_randomgoal
        if self.config['actor_p_trajgoal'] == 1.0:
            # Use only future goals from trajectory
            batch['actor_goals'] = batch['future_goals']
        elif self.config['actor_p_randomgoal'] == 1.0:
            # Use only random goals
            batch['actor_goals'] = batch['random_goals']
        else:
            # Mix them
            use_future = np.random.rand(batch_size) < (
                self.config['actor_p_trajgoal'] / 
                (self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'])
            )
            batch['actor_goals'] = np.where(
                use_future[:, None],
                batch['future_goals'],
                batch['random_goals']
            )
        
        # 6. Compute masks and rewards (for compatibility, not used in TD-InfoNCE critic)
        successes = (idxs == future_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        
        # 7. Apply augmentation if needed
        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    ['observations', 'next_observations', 
                     'future_goals', 'intermediate_future_goals', 
                     'random_goals', 'value_goals', 'actor_goals']
                )
        
        return batch


@dataclasses.dataclass
class ReachabilityGCDataset(GCDataset):
    """
    Optimized dataset for reachability estimator training that inherits from GCDataset.
    Fully vectorized sampling with no for loops for maximum efficiency.
    
    Treats observations directly as goals (no goal_fn abstraction).
    
    Supports two initialization modes:
    1. From trajectories: ReachabilityGCDataset(trajectories=[...], ...)
    2. From Dataset object: ReachabilityGCDataset(dataset=Dataset(...), config={...}, ...)
    """
    
    # Override parent fields
    dataset: Optional[Dataset] = None
    config: Optional[Any] = None
    preprocess_frame_stack: bool = False
    
    # Reachability-specific fields
    trajectories: Optional[Sequence[np.ndarray]] = None
    pi_data_type: str = 'gc'  # One of ['gc', 'hgc', 'td_infonce']
    task_goals: Optional[np.ndarray] = None  # If specified, use these as self_goals

    def __post_init__(self):
        """Initialize the reachability dataset from trajectories or Dataset object."""
        # Check if initialized with Dataset object or trajectories
        if self.dataset is not None and self.trajectories is None:
            # Mode 1: Initialized with Dataset object
            if self.config is None:
                raise ValueError("config must be provided when initializing with a Dataset object")
            
            # Initialize parent class first
            super().__post_init__()
            
            # Reconstruct trajectories from the dataset using terminal flags
            trajectories = []
            current_traj = []
            
            for i in range(self.dataset.size):
                obs = self.dataset['observations'][i]
                current_traj.append(obs)
                
                if self.dataset['terminals'][i]:
                    trajectories.append(np.stack(current_traj, axis=0))
                    current_traj = []
            
            if current_traj:
                trajectories.append(np.stack(current_traj, axis=0))
            
            self.trajectories = trajectories
            
        elif self.trajectories is not None:
            # Mode 2: Initialized with trajectories
            if len(self.trajectories) == 0:
                raise ValueError("At least one trajectory is required.")
            
            # Convert trajectories to flat dataset format
            states_list = []
            next_states_list = []
            terminals_list = []
            
            for traj in self.trajectories:
                traj = np.asarray(traj, dtype=np.float32)
                if traj.ndim != 2 or traj.shape[0] < 2:
                    continue
                
                for i in range(traj.shape[0] - 1):
                    states_list.append(traj[i])
                    next_states_list.append(traj[i + 1])
                    terminals_list.append(i == traj.shape[0] - 2)
            
            if not states_list:
                raise ValueError("No valid trajectories with at least two states.")
            
            # Create Dataset object
            observations = np.stack(states_list, axis=0).astype(np.float32)
            next_observations = np.stack(next_states_list, axis=0).astype(np.float32)
            terminals = np.array(terminals_list, dtype=bool)
            
            self.dataset = Dataset.create(
                observations=observations,
                next_observations=next_observations,
                terminals=terminals,
                freeze=False,
            )
            
            # Create default config if not provided
            if self.config is None:
                self.config = {
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
                }
            
            # Initialize parent class
            super().__post_init__()
        
        else:
            raise ValueError("Either 'dataset' and 'config', or 'trajectories' must be provided")
        
        # Store trajectories and compute metadata
        self.trajectories = [np.asarray(traj, dtype=np.float32) for traj in self.trajectories if traj.shape[0] >= 2]
        self.num_trajectories = len(self.trajectories)
        self.state_dim = self.trajectories[0].shape[1]
        
        # Get observations from dataset
        observations = self.dataset['observations']
        next_observations = self.dataset.get('next_observations')
        if next_observations is None:
            next_observations = self.dataset['observations'][
                np.minimum(np.arange(self.dataset.size) + 1, self.dataset.size - 1)
            ]
        
        # Build mapping from flat indices to trajectory/step indices
        self._traj_ids_np = np.zeros(self.dataset.size, dtype=np.int32)
        self._step_ids_np = np.zeros(self.dataset.size, dtype=np.int32)
        
        flat_idx = 0
        for traj_id, traj in enumerate(self.trajectories):
            for step in range(traj.shape[0] - 1):
                self._traj_ids_np[flat_idx] = traj_id
                self._step_ids_np[flat_idx] = step
                flat_idx += 1
        
        # Precompute trajectory metadata
        self._traj_lengths = np.array([traj.shape[0] for traj in self.trajectories], dtype=np.int32)
        
        # Create padded trajectory array for vectorized indexing
        max_traj_len = max(traj.shape[0] for traj in self.trajectories)
        self._padded_trajectories = np.zeros((self.num_trajectories, max_traj_len, self.state_dim), dtype=np.float32)
        for i, traj in enumerate(self.trajectories):
            self._padded_trajectories[i, :traj.shape[0]] = traj
        
        # Precompute goal pools
        self._states_np = observations.copy()
        self._next_states_np = next_observations.copy()
        
        # Self goals: task goals if provided, otherwise final state of each trajectory
        if self.task_goals is not None:
            if len(self.task_goals) != self.num_trajectories:
                raise ValueError(f"task_goals length ({len(self.task_goals)}) must match number of trajectories ({self.num_trajectories})")
            self._final_goals = np.asarray(self.task_goals, dtype=np.float32)
        else:
            self._final_goals = np.stack([traj[-1] for traj in self.trajectories], axis=0)
        
        # Map each state to its trajectory's final goal
        self._self_goals = self._final_goals[self._traj_ids_np]
        
        # Start states for evaluation
        self._start_states = np.stack([traj[0] for traj in self.trajectories], axis=0).astype(np.float32)
        
        # State bounds
        self._state_min = self._states_np.min(axis=0)
        self._state_max = self._states_np.max(axis=0)
        
        # Create policy dataset
        if self.pi_data_type == 'gc':
            self.pi_dataset = GCDataset(
                dataset=self.dataset,
                config=self.config,
                preprocess_frame_stack=self.preprocess_frame_stack
            )
        elif self.pi_data_type == 'hgc':
            if 'subgoal_steps' not in self.config:
                self.config['subgoal_steps'] = 10
            self.pi_dataset = HGCDataset(
                dataset=self.dataset,
                config=self.config,
                preprocess_frame_stack=self.preprocess_frame_stack
            )
        elif self.pi_data_type == 'td_infonce':
            self.pi_dataset = TDInfoNCEDataset(
                dataset=self.dataset,
                config=self.config,
                preprocess_frame_stack=self.preprocess_frame_stack
            )
        else:
            raise ValueError(f"Invalid pi_data_type: {self.pi_data_type}. Must be one of ['gc', 'hgc', 'td_infonce']")

    def __len__(self) -> int:
        return self._states_np.shape[0]

    @property
    def state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._state_min, self._state_max

    @property
    def start_states(self) -> np.ndarray:
        return self._start_states

    def sample_batch(
        self, 
        batch_size: int,
        num_goals_per_state: int = 4,
        max_skip_horizon: Optional[int] = None,
        num_skip_states: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Sample minibatch with multi-step skip consistency (fully vectorized).
        
        Args:
            batch_size: Number of samples in the batch.
            num_goals_per_state: Number of unlabeled goals per state.
            max_skip_horizon: Maximum horizon for skip states (if None, use trajectory end).
            num_skip_states: Number of skip states to sample per state.
        
        Returns:
            Dictionary with 'reachability' and 'policy' batches.
        """
        # Sample random indices
        idx = np.random.randint(0, len(self), size=batch_size)
        states = self._states_np[idx]
        traj_ids = self._traj_ids_np[idx]
        step_ids = self._step_ids_np[idx]
        
        # === VECTORIZED SKIP STATES SAMPLING ===
        traj_lengths = self._traj_lengths[traj_ids]
        
        # Compute max horizons for each sample
        max_horizons = traj_lengths - step_ids - 1
        if max_skip_horizon is not None:
            max_horizons = np.minimum(max_horizons, max_skip_horizon)
        max_horizons = np.maximum(max_horizons, 1)
        
        # Sample random horizons: [B, M]
        random_vals = np.random.rand(batch_size, num_skip_states)
        horizons = (random_vals * max_horizons[:, None]).astype(np.int32) + 1
        horizons = np.minimum(horizons, max_horizons[:, None])
        
        # Compute skip step indices and gather: [B, M, state_dim]
        skip_step_indices = step_ids[:, None] + horizons
        skip_step_indices = np.minimum(skip_step_indices, (traj_lengths - 1)[:, None])
        skip_states = self._padded_trajectories[traj_ids[:, None], skip_step_indices]
        
        # === VECTORIZED POSITIVE GOALS SAMPLING ===
        # Sample future steps uniformly between (step + 1) and (traj_length - 1)
        random_offsets = np.random.rand(batch_size)
        future_range = traj_lengths - step_ids - 1
        future_range = np.maximum(future_range, 1)
        
        future_steps = step_ids + 1 + (random_offsets * future_range).astype(np.int32)
        future_steps = np.minimum(future_steps, traj_lengths - 1)
        
        # Gather positive goals: [B, state_dim]
        positive_goals = self._padded_trajectories[traj_ids, future_steps]
        
        # === VECTORIZED UNLABELED GOALS VIA ROLLING ===
        # Roll positive_goals to create unlabeled goals (simpler than random sampling)
        # Create multiple rolls to get num_goals_per_state unlabeled goals
        unlabeled_goals = np.stack([
            np.roll(positive_goals, shift=i+1, axis=0) 
            for i in range(num_goals_per_state)
        ], axis=1)  # [B, K, state_dim]
        
        # === SELF GOALS ===
        self_goals = self._self_goals[idx]
        
        # Reachability batch
        reachability_batch = {
            "states": states,
            "skip_states": skip_states,
            "positive_goals": positive_goals,
            "unlabeled_goals": unlabeled_goals,
            "self_goals": self_goals,
            "traj_ids": traj_ids,
        }
        
        # Sample from policy dataset using the same indices
        pi_batch = self.pi_dataset.sample(batch_size, idxs=idx, evaluation=False)
        
        return {
            'reachability': reachability_batch,
            'policy': pi_batch
        }

    def get_anchor_state_and_all_goals(
        self, 
        anchor_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get one anchor state and all goal positions for visualization.
        
        Args:
            anchor_idx: Index of anchor state. If None, use first state of random trajectory.
        
        Returns:
            Tuple of (anchor_state, all_goals, all_states).
        """
        if anchor_idx is None:
            anchor_idx = np.random.randint(0, self.num_trajectories)
            anchor_state = self.trajectories[anchor_idx][0]
        else:
            anchor_state = self._states_np[anchor_idx]
        
        all_goals = self._states_np  # All states can serve as goals
        all_states = self._states_np
        
        return anchor_state.astype(np.float32), all_goals.astype(np.float32), all_states.astype(np.float32)