import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field



import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Optional, Sequence


class RWSValue(nn.Module):
    """Value network that predicts cost-to-go (steps) instead of reachability.
    
    Outputs R(s,g) ∈ [0, H_max] representing predicted minimum steps from s to g.
    Lower is better (cost function).
    
    Uses H_max * sigmoid(z) to ensure output is in [0, H_max].
    """
    hidden_dims: Sequence[int]
    h_max: float = 100.0
    layer_norm: bool = True
    ensemble: bool = False
    gc_encoder: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, observations, goals):
        """Predict step costs from observations to goals.
        
        Args:
            observations: [B, obs_dim]
            goals: [B, goal_dim]
        
        Returns:
            Step costs in [0, H_max], shape [B, 1]
        """
        # Encode observations and goals
        if self.gc_encoder is not None:
            x = self.gc_encoder(observations, goals)
        else:
            x = jnp.concatenate([observations, goals], axis=-1)
        
        # MLP backbone
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f'fc{i}')(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        # Output layer: raw logits
        logits = nn.Dense(1, name='fc_out')(x)
        
        # Map to [0, H_max] using scaled sigmoid
        v = nn.sigmoid(logits)
        
        return v

class RWSCritic(nn.Module):
    """Q-Critic network that predicts V(s', g) given (s, a, g).
    
    Takes state, action, and goal as input and predicts the value of the next state.
    Similar architecture to RWSValue but conditioned on actions.
    
    Outputs Q(s,a,g) ≈ V(s',g) representing predicted value after taking action a.
    """
    hidden_dims: Sequence[int]
    h_max: float = 100.0
    layer_norm: bool = True
    gc_encoder: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, observations, goals, actions):
        """Predict next state value given current state, action, and goal.
        
        Args:
            observations: [B, obs_dim]
            goals: [B, goal_dim]
            actions: [B, action_dim]
        
        Returns:
            Predicted V(s', g) values, shape [B, 1]
        """
        # Encode observations and goals
        if self.gc_encoder is not None:
            obs_goal_features = self.gc_encoder(observations, goals)
            # Concatenate action features after encoding
            x = jnp.concatenate([obs_goal_features, actions], axis=-1)
        else:
            # Concatenate all inputs
            x = jnp.concatenate([observations, goals, actions], axis=-1)
        
        # MLP backbone
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f'fc{i}')(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        
        # Output layer: raw logits
        logits = nn.Dense(1, name='fc_out')(x)
        
        # Map to [0, H_max] using scaled sigmoid (same as value network)
        q = nn.sigmoid(logits)
        
        return q


class RWSAgent(flax.struct.PyTreeNode):
    """Reachability estimator agent using the loss from rws_test.py.
    
    This agent trains a value network to predict reachability between states and goals
    using PU-RANK loss and multi-step consistency loss with a target network.
    
    Additionally trains a Q critic that approximates V(s', g) for action-conditioned predictions.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        """Compute Q critic loss: MSE(Q(s,a,g), V(s',g))
        
        The Q critic learns to predict the value of the next state V(s', g) 
        given current state, action, and goal.
        
        Args:
            batch: Dictionary containing:
                - states: [B, state_dim]
                - actions: [B, action_dim]
                - skip_states: [B, M, state_dim] - we use skip_states[:,0,:] as next states
                - positive_goals: [B, goal_dim]
            grad_params: Parameters to compute gradients for
        
        Returns:
            Tuple of (critic_loss, info_dict)
        """
        states = batch['states']  # [B, state_dim]
        actions = batch['actions']  # [B, action_dim]
        next_states = batch['skip_states'][:, 0, :]  # [B, state_dim] - first skip state as next state
        goals = batch['positive_goals']  # [B, goal_dim]
        
        # Q(s, a, g) prediction
        q_pred = self.network.select('critic')(states, goals, actions, params=grad_params)  # [B, 1]
        q_pred = jnp.squeeze(q_pred, axis=-1)  # [B]
        
        # V(s', g) target (stop gradient to prevent backprop through value network)
        v_target = self.network.select('value')(next_states, goals)  # [B, 1]
        v_target = jnp.squeeze(v_target, axis=-1)  # [B]
        v_target = jax.lax.stop_gradient(v_target)
        
        # MSE loss
        critic_loss = jnp.mean((q_pred - v_target) ** 2)
        
        info = {
            'critic_loss': critic_loss,
            'q_pred_mean': jnp.mean(q_pred),
            'v_target_mean': jnp.mean(v_target),
            'q_v_diff_mean': jnp.mean(jnp.abs(q_pred - v_target)),
        }
        
        return critic_loss, info

    def reachability_loss(self, batch, grad_params):
        """Solution E: TD-style monotone inequality (no Q) + pairwise ranking.

        Args:
            batch: Dictionary containing:
                - states: [B, state_dim]
                - skip_states: [B, M, state_dim]   # future/intermediate states of each anchor state
                - positive_goals: [B, goal_dim]
                - unlabeled_goals: [B, K, goal_dim]
            grad_params: Parameters to compute gradients for (current/online net).

        Returns:
            Tuple of (total_loss, info_dict)
        """
        import jax
        import jax.numpy as jnp

        states        = batch['states']          # [B, state_dim]
        skip_states   = batch['skip_states']     # [B, M, state_dim]
        pos_goals     = batch['positive_goals']  # [B, goal_dim]
        unl_goals     = batch['unlabeled_goals'] # [B, K, goal_dim]

        B = states.shape[0]
        M = skip_states.shape[1]
        K = unl_goals.shape[1]

        gamma = self.config.get('discount', 0.99)

        margin       = self.config.get('rank_margin', 0.001)
        lambda_mono  = self.config.get('lambda_mono', self.config.get('lambda_cons', 1.0))  # fallback to old key

        # === Gamma weights for skip states [gamma^1, gamma^2, ..., gamma^M] ===
        gamma_weights = gamma ** jnp.arange(1, M + 1)  # [M]
        gamma_weights = gamma_weights.reshape(1, M)     # [1, M] for broadcasting

        # === Current predictions f_theta(s, g) ===
        # Positives
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1] or [B]
        pred_pos = jnp.squeeze(pred_pos, axis=-1)                                       # [B]

        # Unlabeled (vectorized over K)
        states_exp = jnp.expand_dims(states, 1)                 # [B, 1, state_dim]
        states_exp = jnp.broadcast_to(states_exp, (B, K, states.shape[-1]))  # [B, K, state_dim]
        pred_unl = self.network.select('value')(
            states_exp.reshape(B*K, -1),
            unl_goals.reshape(B*K, -1),
            params=grad_params
        )
        pred_unl = jnp.squeeze(pred_unl, axis=-1).reshape(B, K)   # [B, K]

        # === Pairwise ranking loss: positives vs ALL unlabeled (prior-free, margin-based) ===
        # logits_ij = f(s_i, g^+_i) - f(s_i, g~_{i,j}) - margin
        pos_exp = jnp.expand_dims(pred_pos, 1)           # [B, 1]
        rank_logits = pos_exp - pred_unl - margin        # [B, K]
        # logistic pairwise loss: log(1 + exp(-logit))
        rank_loss = jnp.mean(jax.nn.softplus(-rank_logits))

        # === TD-style monotone inequality against target net (no Q): f(s, g) >= max_m [gamma^m * f_bar(s_mid^m, g)] ===
        # 1) Positives: compute target on skip states for each (s_i, g^+_i), weight by gamma^m, then max over M
        skip_flat = skip_states.reshape(B*M, -1)           # [B*M, state_dim]
        pos_goals_exp = jnp.repeat(jnp.expand_dims(pos_goals, 1), M, axis=1).reshape(B*M, -1)  # [B*M, goal_dim]

        target_pos_all = self.network.select('target_value')(skip_flat, pos_goals_exp)  # [B*M, 1] or [B*M]
        target_pos_all = jnp.squeeze(target_pos_all, axis=-1).reshape(B, M)             # [B, M]
        
        # Apply gamma weighting to each skip state
        target_pos_all_weighted = target_pos_all * gamma_weights  # [B, M]
        target_pos_max = jnp.max(target_pos_all_weighted, axis=1)  # [B]

        # Monotone penalty for positives: relu(target_max - current)
        mono_pos = jax.nn.relu(jax.lax.stop_gradient(target_pos_max) - pred_pos) # [B]
        mono_pos = jnp.mean(mono_pos)

        # 2) Unlabeled: compute target on skip states for each (s_i, g~_{i,j}), weight by gamma^m, then max over M
        # Expand (s_mid, g_unl) pairs -> [B, M, K, ...] flattened once
        skip_exp_unl = jnp.expand_dims(skip_states, 2)                 # [B, M, 1, state_dim]
        skip_exp_unl = jnp.broadcast_to(skip_exp_unl, (B, M, K, skip_states.shape[-1]))  # [B, M, K, state_dim]
        skip_exp_unl = skip_exp_unl.reshape(B*M*K, -1)                  # [B*M*K, state_dim]

        unl_goals_exp = jnp.expand_dims(unl_goals, 1)                   # [B, 1, K, goal_dim]
        unl_goals_exp = jnp.broadcast_to(unl_goals_exp, (B, M, K, unl_goals.shape[-1]))
        unl_goals_exp = unl_goals_exp.reshape(B*M*K, -1)                # [B*M*K, goal_dim]

        target_unl_all = self.network.select('target_value')(skip_exp_unl, unl_goals_exp)  # [B*M*K, 1] or [B*M*K]
        target_unl_all = jnp.squeeze(target_unl_all, axis=-1).reshape(B, M, K)             # [B, M, K]
        
        # Apply gamma weighting to each skip state
        target_unl_all_weighted = target_unl_all * gamma_weights.reshape(1, M, 1)  # [B, M, K]
        target_unl_max = jnp.max(target_unl_all_weighted, axis=1)  # [B, K]

        # Monotone penalty for unlabeled
        pred_unl_exp = jnp.expand_dims(pred_unl, 1)  # [B, 1, K] for broadcasting
        mono_unl = jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl)  # [B, K]
        mono_unl = jnp.mean(mono_unl)

        # === Combine losses ===
        total_loss = rank_loss + lambda_mono * (mono_pos + mono_unl)

        info = {
            'loss': total_loss,
            'rank_loss': rank_loss,
            'mono_pos': mono_pos,
            'mono_unl': mono_unl,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'tgt_pos_max_mean': jnp.mean(target_pos_max),
            'tgt_unl_max_mean': jnp.mean(target_unl_max),
        }
        return total_loss, info


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (reachability + critic)."""
        info = {}
        
        # Reachability loss (value network)
        reach_loss, reach_info = self.reachability_loss(batch, grad_params)
        for k, v in reach_info.items():
            info[f'value/{k}'] = v
        
        # Critic loss (Q network) - only if actions are in batch
        if 'actions' in batch:
            critic_loss, critic_info = self.critic_loss(batch, grad_params)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v
            
            # Weight for critic loss
            lambda_critic = self.config.get('lambda_critic', 1.0)
            total_loss = reach_loss + lambda_critic * critic_loss
        else:
            total_loss = reach_loss
        
        return total_loss, info

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
    
    @jax.jit
    def predict_q_value(self, observations, goals, actions):
        """Predict Q values for state-action-goal triplets.
        
        Args:
            observations: [B, obs_dim] or [obs_dim]
            goals: [B, goal_dim] or [goal_dim]
            actions: [B, action_dim] or [action_dim]
        
        Returns:
            Q values approximating V(s', g), shape [B, 1] or [1]
        """
        return self.network.select('critic')(observations, goals, actions)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new reachability agent with Q critic.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
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
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # Define reachability value network (with sigmoid output)
        value_def = RWSValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )
        
        # Define Q critic network (approximates V(s', g))
        # Use RWSCritic that takes (s, a, g) as input
        critic_def = RWSCritic(
            hidden_dims=config.get('critic_hidden_dims', config['value_hidden_dims']),
            layer_norm=config['layer_norm'],
            gc_encoder=encoders.get('critic'),
        )

        # Create network with value, target_value, and critic
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
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
            critic_hidden_dims=(256, 256, 256),  # Q critic network hidden dimensions
            layer_norm=True,  # Whether to use layer normalization
            tau=0.995,  # Target network EMA update rate (higher = slower update)
            
            # Reachability loss hyperparameters
            rank_margin=-0.05,  # Margin for ranking loss
            lambda_cons=1.0,  # Weight for consistency loss
            lambda_critic=1.0,  # Weight for critic loss
            
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