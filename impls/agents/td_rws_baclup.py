import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
#from utils.networks import StepsValue  # New network that outputs steps, not reachability


import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Optional, Sequence


class StepsValue(nn.Module):
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
        # sigmoid(z) ∈ [0, 1], so H_max * sigmoid(z) ∈ [0, H_max]
        steps = self.h_max * nn.sigmoid(logits)
        
        return steps



class TDRWSAgent(flax.struct.PyTreeNode):
    """Step-Based Cost-to-Go Reachability Agent.
    
    Learns R(s,g) ∈ [0, H_max] = predicted minimum steps from s to g.
    Lower is better (cost function, not reward).
    
    Loss components:
    1. PU-RANK: Positive costs < unlabeled costs (by margin)
    2. POSITIVE UPPER-BOUND: R(s,g+) ≤ d+ (known steps)
    3. UNLABELED DP UPPER-BOUND: R(s,g) ≤ min_h{min(R(s,s_{t+h}),h) + R(s_{t+h},g)}
    
    This is the CORRECTED version with per-goal DP bounds.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def reachability_loss_v0(self, batch, grad_params):
        """
        Step-based losses:
          - PU rank: pos < unl (hard-negative by min unlabeled)
          - Pos upper-bound: R(s,g+) <= d+
          - Unl DP upper-bound: R(s,g_k) <= min_h { min(R_tgt(s, s_{t+h}), h) + R_tgt(s_{t+h}, g_k) }
        
        Args:
            batch: Dictionary containing:
                - states: [B, state_dim]
                - skip_states: [B, M, state_dim]
                - positive_goals: [B, goal_dim]
                - unlabeled_goals: [B, K, goal_dim]
                - positive_steps: [B] - number of steps to positive goal
                - skip_steps: [B, M] - number of steps to each skip state
            grad_params: Parameters to compute gradients for.
        
        Returns:
            Tuple of (total_loss, info_dict)
        """
        states      = batch['states']           # [B, Ds]
        skip_states = batch['skip_states']      # [B, M, Ds]
        pos_goals   = batch['positive_goals']   # [B, Dg]
        unl_goals   = batch['unlabeled_goals']  # [B, K, Dg]
        pos_steps   = batch['positive_steps']   # [B]
        skip_steps  = batch['skip_steps']       # [B, M]

        B, M = skip_states.shape[:2]
        K    = unl_goals.shape[1]

        # ---- Predictions R(s, g) ----
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1]
        
        # R(s, g_unl_k) for all unlabeled goals
        states_K = jnp.repeat(states[:, None, :], repeats=K, axis=1).reshape(B*K, -1)
        goals_K = unl_goals.reshape(B*K, -1)
        pred_unl = self.network.select('value')(states_K, goals_K, params=grad_params).reshape(B, K)  # [B, K]

        # ---- 1) PU-RANK: enforce pos + m <= min_unl ----
        # Since R is a cost (lower is better), positives should be smaller than unlabeled
        #min_unl = jnp.min(pred_unl, axis=1, keepdims=True)  # [B, 1] hardest unlabeled
        #rank_logits = min_unl - pred_pos - self.config['rank_margin']  # want >= 0
        #rank_loss = jnp.mean(jax.nn.softplus(-rank_logits))
        mean_unl = jnp.mean(pred_unl, axis=1, keepdims=True)        # [B, 1]
        rank_logits = mean_unl - pred_pos - self.config['rank_margin']                         # want >= 0
        #rank_loss   = jnp.mean(-mean_unl) / self.config['h_max']
        rank_loss   = jnp.mean(-rank_logits) / self.config['h_max']
        #rank_loss   = jnp.mean(jax.nn.softplus(-rank_logits))

        # ---- 2) POSITIVE UPPER-BOUND: R(s,g+) <= d+ ----
        # We know the true steps, so prediction shouldn't exceed it
        d_pos = pos_steps[:, None]  # [B, 1]
        hinge_pos = jax.nn.relu(pred_pos - d_pos)  # only penalize if above true steps
        loss_pos = jnp.mean(jnp.square(hinge_pos / (self.config['h_max']) )) 

        # ---- 3) UNLABELED DP UPPER-BOUND (per-goal) ----
        
        # 3a) R_tgt(s, s_{t+h}): cost from current state to skip states
        sM_states = jnp.repeat(states[:, None, :], repeats=M, axis=1).reshape(B*M, -1)
        sM_skips = skip_states.reshape(B*M, -1)
        cost_s_to_skip = self.network.select('target_value')(sM_states, sM_skips).reshape(B, M)  # [B, M]
        
        # Admissible upper bound to waypoint: min(predicted, observed h)
        # This keeps the bound tight while allowing the model to learn shorter paths
        steps_to_skip = jnp.minimum(cost_s_to_skip, skip_steps)  # [B, M]
        #steps_to_skip = skip_steps  # [B, M]

        # 3b) R_tgt(s_{t+h}, g_k) for all (skip_state, goal) pairs
        # Expand to [B, M, K, *]
        skips_MK = jnp.repeat(skip_states[:, :, None, :], repeats=K, axis=2).reshape(B*M*K, -1)
        goals_MK = jnp.repeat(unl_goals[:, None, :, :], repeats=M, axis=1).reshape(B*M*K, -1)
        cost_skip_to_goal = self.network.select('target_value')(skips_MK, goals_MK).reshape(B, M, K)  # [B, M, K]

        # 3c) DP upper bound per goal: min over h of {steps_to_skip[h] + cost_skip_to_goal[h,k]}
        # For each goal k, we compute the best path through all skip states
        # steps_to_skip: [B, M] -> [B, M, 1]
        # cost_skip_to_goal: [B, M, K]
        # sum: [B, M, K]
        # min over M: [B, K]
        ub_unl = jnp.min(steps_to_skip[:, :, None] + cost_skip_to_goal, axis=1)  # [B, K]

        # 3d) One-sided squared hinge: enforce R(s,g_k) <= ub_unl
        hinge_unl = pred_unl - jax.lax.stop_gradient(ub_unl)
        loss_unl = jnp.mean(jnp.square(hinge_unl / (self.config['h_max']) )) 

        # ---- TOTAL LOSS ----
        total = (
            rank_loss
            + self.config['lambda_cons_pos'] * loss_pos
            + self.config['lambda_cons_unl'] * loss_unl
        )

        info = {
            'loss_total': total,
            'loss_rank': rank_loss,
            'loss_cons_pos': loss_pos,
            'loss_cons_unl': loss_unl,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'min_unl_mean': jnp.mean(mean_unl),
            'd_pos_mean': jnp.mean(d_pos),
            'cost_s_to_skip_mean': jnp.mean(cost_s_to_skip),
            'steps_to_skip_mean': jnp.mean(steps_to_skip),
            'cost_skip_to_goal_mean': jnp.mean(cost_skip_to_goal),
            'ub_unl_mean': jnp.mean(ub_unl),
            'hinge_pos_mean': jnp.mean(hinge_pos),
            'hinge_unl_mean': jnp.mean(hinge_unl),
        }
        
        return total, info

    
    def reachability_loss(self, batch, grad_params):
        """
        Clean 1-step Bellman sandwich loss.
        
        Loss components:
        1. Upper bound hinge: R(s_t,g) ≤ 1 + R_target(s_{t+1},g)
        2. Lower bound hinge: R(s_t,g) ≥ max(0, R_target(s_{t+1},g) - 1)
        3. Identity anchor: R(s,s) = 0
        4. Edge anchor: R(s_t, s_{t+1}) = 1
        5. Tiny shrinkage: minimize R(s,g)
        
        Args:
            batch: Dictionary containing:
                - states: [B, state_dim] - current states s_t
                - skip_states: [B, M, state_dim] - skip states (we use first for s_{t+1})
                - positive_goals: [B, goal_dim] - positive goals
                - unlabeled_goals: [B, K, goal_dim] - unlabeled goals
            grad_params: Parameters to compute gradients for.
        
        Returns:
            Tuple of (total_loss, info_dict)
        """
        states = batch['states']  # [B, Ds] - s_t
        skip_states = batch['skip_states']  # [B, M, Ds]
        pos_goals = batch['positive_goals']  # [B, Dg]
        unl_goals = batch['unlabeled_goals']  # [B, K, Dg]

        B, M = skip_states.shape[:2]
        K = unl_goals.shape[1]

        # Extract 1-step next states (first skip state)
        next_states = skip_states[:, 0, :]  # [B, Ds] - s_{t+1}

        # Combine all goals for efficiency: [positive] + [unlabeled]
        all_goals = jnp.concatenate([pos_goals[:, None, :], unl_goals], axis=1)  # [B, 1+K, Dg]
        num_goals = 1 + K

        # ===== 1. Bellman Sandwich Constraints =====
        
        # Expand states and next_states for all goals
        states_rep = jnp.repeat(states[:, None, :], repeats=num_goals, axis=1).reshape(B * num_goals, -1)
        next_states_rep = jnp.repeat(next_states[:, None, :], repeats=num_goals, axis=1).reshape(B * num_goals, -1)
        goals_flat = all_goals.reshape(B * num_goals, -1)

        # Current predictions R_θ(s_t, g)
        if self.config.get('use_ensemble', True):
            r1, r2 = self.network.select('value')(states_rep, goals_flat, params=grad_params)
            r_pred = (r1 + r2) / 2  # [B*(1+K), 1]
        else:
            r_pred = self.network.select('value')(states_rep, goals_flat, params=grad_params)
        r_pred = r_pred.reshape(B, num_goals)  # [B, 1+K]

        # Target predictions R_target(s_{t+1}, g)
        if self.config.get('use_ensemble', True):
            r1_next, r2_next = self.network.select('target_value')(next_states_rep, goals_flat)
            r_next = jnp.minimum(r1_next, r2_next)  # Min for ensemble (more conservative)
        else:
            r_next = self.network.select('target_value')(next_states_rep, goals_flat)
        r_next = r_next.reshape(B, num_goals)  # [B, 1+K]

        # Upper bound: R(s_t, g) ≤ 1 + R_target(s_{t+1}, g)
        ub_target = 1.0 + jax.lax.stop_gradient(r_next)
        ub_violation = r_pred - ub_target
        ub_hinge = jax.nn.relu(ub_violation)  # Only penalize violations
        loss_ub = jnp.mean(jnp.square(ub_hinge))

        # Lower bound: R(s_t, g) ≥ max(0, R_target(s_{t+1}, g) - 1)
        lb_target = jnp.maximum(0.0, jax.lax.stop_gradient(r_next) - 1.0)
        lb_violation = lb_target - r_pred
        lb_hinge = jax.nn.relu(lb_violation)  # Only penalize violations
        loss_lb = jnp.mean(jnp.square(lb_hinge))

        # ===== 2. Identity Anchor: R(s, s) = 0 =====
        
        # Predict self-distance
        if self.config.get('use_ensemble', True):
            r1_id, r2_id = self.network.select('value')(states, states, params=grad_params)
            r_identity = (r1_id + r2_id) / 2
        else:
            r_identity = self.network.select('value')(states, states, params=grad_params)
        
        loss_identity = jnp.mean(jnp.square(r_identity))  # Should be 0

        # ===== 3. Edge Anchor: R(s_t, s_{t+1}) = 1 =====
        
        # Predict 1-step edge cost
        if self.config.get('use_ensemble', True):
            r1_edge, r2_edge = self.network.select('value')(states, next_states, params=grad_params)
            r_edge = (r1_edge + r2_edge) / 2
        else:
            r_edge = self.network.select('value')(states, next_states, params=grad_params)
        
        loss_edge = jnp.mean(jnp.square(r_edge - 1.0))  # Should be 1

        # ===== 4. Shrinkage: Minimize R(s,g) =====
        
        # Tiny coefficient to prefer minimal feasible solution
        loss_shrink = jnp.mean(r_pred)

        # ===== Total Loss =====
        
        lambda_ub = self.config.get('lambda_ub', 1.0)
        lambda_lb = self.config.get('lambda_lb', 1.0)
        lambda_id = self.config.get('lambda_identity', 10.0)
        lambda_edge = self.config.get('lambda_edge', 10.0)
        epsilon = self.config.get('epsilon_shrink', 0.01)

        total = (
            lambda_ub * loss_ub
            + lambda_lb * loss_lb
            + lambda_id * loss_identity
            + lambda_edge * loss_edge
            + epsilon * loss_shrink
        )

        info = {
            'loss_total': total,
            'loss_ub': loss_ub,
            'loss_lb': loss_lb,
            'loss_identity': loss_identity,
            'loss_edge': loss_edge,
            'loss_shrink': loss_shrink,
            'r_pred_mean': jnp.mean(r_pred),
            'r_next_mean': jnp.mean(r_next),
            'r_identity_mean': jnp.mean(r_identity),
            'r_edge_mean': jnp.mean(r_edge),
            'ub_violation_mean': jnp.mean(ub_hinge),
            'lb_violation_mean': jnp.mean(lb_hinge),
            'ub_target_mean': jnp.mean(ub_target),
            'lb_target_mean': jnp.mean(lb_target),
        }

        return total, info

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
    def predict_steps(self, observations, goals):
        """Predict step costs between observations and goals.
        
        Args:
            observations: [B, obs_dim] or [obs_dim]
            goals: [B, goal_dim] or [goal_dim]
        
        Returns:
            Step costs in [0, H_max], shape [B, 1] or [1]
        """
        return self.network.select('value')(observations, goals)
    
    @jax.jit
    def predict_reachability(self, observations, goals):
        """Predict reachability as normalized inverse cost (for compatibility).
        
        Args:
            observations: [B, obs_dim] or [obs_dim]
            goals: [B, goal_dim] or [goal_dim]
        
        Returns:
            Reachability scores in [0, 1], shape [B, 1] or [1]
            (computed as 1 - steps/H_max)
        """
        steps = self.predict_steps(observations, goals)
        return 1.0 - steps / self.config['h_max']

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,  # Not used, but kept for compatibility
        config,
    ):
        """Create a new step-based TD-RWS agent.

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

        # Define step-cost value network (outputs steps in [0, H_max])
        value_def = StepsValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            h_max=config['h_max'],
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
            agent_name='td_rws',  # Agent name
            lr=3e-4,  # Learning rate
            batch_size=1024,  # Batch size
            value_hidden_dims=(256, 256, 256),  # Step-cost network hidden dimensions
            layer_norm=True,  # Whether to use layer normalization
            tau=0.995,  # Target network EMA update rate (higher = slower update)
            
            # Reachability loss hyperparameters
            rank_margin=0.,  # Margin for ranking loss (in steps)
            lambda_cons_pos=1.0,  # Weight for positive upper-bound loss
            lambda_cons_unl=1.0,  # Weight for unlabeled DP upper-bound loss
            
            # Step-based hyperparameters
            h_max=2000.0,  # Maximum step horizon (output range)
            
            # Dataset hyperparameters
            dataset_class='ReachabilityDataset',  # Dataset class name
            num_goals_per_state=4,  # Number of unlabeled goals per state (K)
            num_skip_states=3,  # Number of skip states per sample (M)
            max_skip_horizon=None,  # Maximum horizon for skip states (None = trajectory end)
            
            # GCDataset config (inherited by ReachabilityGCDataset)
            discount=0.99,  # Not used in step-based, kept for compatibility
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