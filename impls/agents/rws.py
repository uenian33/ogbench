import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import RWSValue


class RWSAgent(flax.struct.PyTreeNode):
    """Reachability estimator agent using the loss from rws_test.py.
    
    This agent trains a value network to predict reachability between states and goals
    using PU-RANK loss and multi-step consistency loss with a target network.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def reachability_loss(self, batch, grad_params):
        """Compute the combined reachability loss (PU-RANK + Consistency).
        
        Args:
            batch: Dictionary containing:
                - states: [B, state_dim]
                - skip_states: [B, M, state_dim]
                - positive_goals: [B, goal_dim]
                - unlabeled_goals: [B, K, goal_dim]
                - self_goals: [B, goal_dim]
            grad_params: Parameters to compute gradients for.
        
        Returns:
            Tuple of (total_loss, info_dict)
        """
        states = batch['states']  # [B, state_dim]
        skip_states = batch['skip_states']  # [B, M, state_dim]
        pos_goals = batch['positive_goals']  # [B, goal_dim]
        unl_goals = batch['unlabeled_goals']  # [B, K, goal_dim]
        
        B = states.shape[0]
        M = skip_states.shape[1]
        K = unl_goals.shape[1]
        
        # === 1. PU-RANK LOSS ===
        # Predict reachability for positive goals
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1]
        
        # Predict reachability for all K unlabeled goals
        # Expand states to [B, K, state_dim]
        states_expanded = jnp.expand_dims(states, 1)  # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1))  # [B, K, state_dim]
        
        # Flatten for batch processing
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        
        # Predict and reshape
        pred_unl = self.network.select('value')(states_flat, unl_goals_flat, params=grad_params)
        pred_unl = pred_unl.reshape(B, K)  # [B, K]
        
        # Ranking loss: positive should be higher than mean of unlabeled
        rank_logits = pred_pos - pred_unl.mean(axis=1, keepdims=True) - self.config['rank_margin']
        rank_loss = jnp.mean(jax.nn.softplus(-rank_logits))
        
        # === 2. MULTI-STEP CONSISTENCY LOSS WITH MAX-POOLING ===
        # For positives: max_i r_target(s_{t+hi}, g^+)
        skip_states_flat = skip_states.reshape(B * M, -1)  # [B*M, state_dim]
        
        # Expand positive goals to match skip states
        pos_goals_expanded = jnp.expand_dims(pos_goals, 1)  # [B, 1, goal_dim]
        pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1))  # [B, M, goal_dim]
        pos_goals_expanded = pos_goals_expanded.reshape(B * M, -1)  # [B*M, goal_dim]
        
        # Use target network (no gradients)
        target_pos_all = self.network.select('target_value')(skip_states_flat, pos_goals_expanded)
        target_pos_all = target_pos_all.reshape(B, M)  # [B, M]
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B, 1]
        
        # Consistency loss for positives
        cons_pos = jnp.mean(jnp.square(pred_pos - jax.lax.stop_gradient(target_pos_max)))
        
        # For unlabeled: max_i r_target(s_{t+hi}, g_unl)
        # Expand skip states for all unlabeled goals: [B, M, K, state_dim]
        skip_states_expanded_unl = jnp.expand_dims(skip_states, 2)  # [B, M, 1, state_dim]
        skip_states_expanded_unl = jnp.tile(skip_states_expanded_unl, (1, 1, K, 1))  # [B, M, K, state_dim]
        skip_states_expanded_unl = skip_states_expanded_unl.reshape(B * M * K, -1)
        
        # Expand unlabeled goals: [B, M, K, goal_dim]
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)  # [B, 1, K, goal_dim]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1))  # [B, M, K, goal_dim]
        unl_goals_expanded = unl_goals_expanded.reshape(B * M * K, -1)
        
        # Compute target predictions
        target_unl_all = self.network.select('target_value')(skip_states_expanded_unl, unl_goals_expanded)
        target_unl_all = target_unl_all.reshape(B, M, K)  # [B, M, K]
        target_unl_max = jnp.max(target_unl_all, axis=1)  # [B, K]
        
        # Consistency loss for unlabeled (only penalize if target > current)
        cons_unl = jnp.mean(jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl))
        
        # Combined consistency loss
        consistency_loss = 0.5 * (cons_pos + cons_unl)
        
        # === TOTAL LOSS ===
        total_loss = rank_loss + self.config['lambda_cons'] * consistency_loss
        
        info = {
            'loss_total': total_loss,
            'loss_rank': rank_loss,
            'loss_cons': consistency_loss,
            'cons_pos': cons_pos,
            'cons_unl': cons_unl,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'target_pos_max_mean': jnp.mean(target_pos_max),
            'target_unl_max_mean': jnp.mean(target_unl_max),
        }
        
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (wrapper for compatibility)."""
        return self.reachability_loss(batch, grad_params)

    def target_update(self, network):
        """Update the target network using EMA."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params['modules_value'],
            network.params['modules_target_value'],
        )
        network.params['modules_target_value'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def predict_reachability(self, observations, goals):
        """Predict reachability scores between observations and goals.
        
        Args:
            observations: [B, obs_dim] or [obs_dim]
            goals: [B, goal_dim] or [goal_dim]
        
        Returns:
            Reachability scores in [0, 1], shape [B, 1] or [1]
        """
        return self.network.select('value')(observations, goals)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,  # Not used, but kept for compatibility
        config,
    ):
        """Create a new reachability agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions (not used, for compatibility).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations  # Goals have same shape as observations

        # Define encoder if specified
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())

        # Define reachability value network (with sigmoid output)
        value_def = RWSValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        # Create network with both value and target_value
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target network with same parameters
        params = network_params
        params['modules_target_value'] = copy.deepcopy(params['modules_value'])

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters
            agent_name='rws',  # Agent name
            lr=3e-4,  # Learning rate
            batch_size=1024,  # Batch size
            value_hidden_dims=(256, 256, 256),  # Reachability network hidden dimensions
            layer_norm=True,  # Whether to use layer normalization
            tau=0.995,  # Target network EMA update rate (higher = slower update)
            
            # Reachability loss hyperparameters
            rank_margin=-0.05,  # Margin for ranking loss
            lambda_cons=1.0,  # Weight for consistency loss
            
            # Dataset hyperparameters
            dataset_class='ReachabilityDataset',  # Dataset class name
            num_goals_per_state=4,  # Number of unlabeled goals per state (K)
            num_skip_states=3,  # Number of skip states per sample (M)
            max_skip_horizon=None,  # Maximum horizon for skip states (None = trajectory end)
            
            # GCDataset config (inherited by ReachabilityGCDataset)
            discount=0.99,  # Discount factor for geometric sampling
            value_p_curgoal=0.0,  # Not used in reachability training
            value_p_trajgoal=1.0,  # Sample positive goals from trajectory
            value_p_randomgoal=0.0,  # Don't use random goals as positives
            value_geom_sample=False,  # Use uniform sampling for future goals
            actor_p_curgoal=0.0,  # Not used
            actor_p_trajgoal=0.5,  # Not used
            actor_p_randomgoal=0.5,  # Not used
            actor_geom_sample=False,  # Not used
            gc_negative=False,  # Not used
            p_aug=None,  # No image augmentation
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder (None for state-based)
        )
    )
    return config