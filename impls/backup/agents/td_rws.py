import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCMRNValue, LogParam



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
    


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function with rational cap.

    v_raw = ||phi_sym(s) - phi_sym(g)||_2 + ReLU(max(phi_asym(s) - phi_asym(g)))
    v = max_H * v_raw / (v_raw + c),  with c>0 (learned via softplus).

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension (will be split in half: sym | asym).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
        max_H: Upper bound for the capped value output.
        c_init: Initial value of c in the rational cap.
        learn_c: Whether to learn c (if False, c is fixed at c_init).
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    # rational-cap hyperparams
    max_H: float = 1000.0
    c_init: float = 100.0
    learn_c: bool = True

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim),
                       activate_final=False, layer_norm=self.layer_norm)
        if self.learn_c:
            # store c in pre-softplus form so softplus(c_raw) ≈ c_init
            raw_init = jnp.log(jnp.expm1(self.c_init))
            self.c_raw = self.param('c_raw',
                                    nn.initializers.constant(raw_init), ())

    def _cap_rational(self, z):
        """max_H * z / (z + c), with c>0 and z>=0."""
        if self.learn_c:
            c = jax.nn.softplus(self.c_raw) + 1e-6
        else:
            c = jnp.asarray(self.c_init)
        z = jnp.maximum(z, 0.0)
        return self.max_H * (z / (z + c))

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the capped MRN value function."""
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        # split into symmetric/asymmetric halves
        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]

        # MRN components
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        euclid = jnp.sqrt(jnp.maximum(squared_dist, 1e-12))
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = euclid + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


    
class TDRWSAgent(flax.struct.PyTreeNode):
    """Reachability estimator agent using Version A: Step-valued cost.
    
    This agent trains a StepsValue network to predict R(s,g) as minimum steps-to-goal
    where LOWER is BETTER. The network outputs values in [0, H_max] using scaled sigmoid.
    
    The loss enforces:
    - Bellman-type upper/lower bounds via multi-step waypoints
    - Anchor constraints (identity, edge costs)
    - PU ranking with cost semantics (positives should be smaller)
    - Hindsight positive upper bound
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def reachability_loss_v0(self, batch, grad_params):
        """Compute the QRL value loss."""
        d_neg = self.network.select('value')(batch['states'], batch['unlabeled_goals'][:,0], params=grad_params)
        d_pos = self.network.select('value')(batch['states'], batch['next_states'], params=grad_params)
        lam = self.network.select('lam')(params=grad_params)

        # Apply loss shaping following the original implementation.
        d_neg_loss = (100 * jax.nn.softplus(5 - d_neg / 100)).mean()
        d_pos_loss = (jax.nn.relu(d_pos - 1) ** 2).mean()

        value_loss = d_neg_loss + d_pos_loss * jax.lax.stop_gradient(lam)
        lam_loss = lam * (0.05 - jax.lax.stop_gradient(d_pos_loss))

        total_loss = value_loss + lam_loss

        return total_loss, {
            'total_loss': total_loss,
            'value_loss': value_loss,
            'lam_loss': lam_loss,
            'd_neg_loss': d_neg_loss,
            'd_neg_mean': d_neg.mean(),
            'd_neg_max': d_neg.max(),
            'd_neg_min': d_neg.min(),
            'd_pos_loss': d_pos_loss,
            'd_pos_mean': d_pos.mean(),
            'd_pos_max': d_pos.max(),
            'd_pos_min': d_pos.min(),
            'lam': lam,
        }
    
    def reachability_loss_v1(self, batch, grad_params):
        """Compute the combined reachability loss (Version A: Step-valued cost).
        
        Version A treats R(s,g) as minimum steps-to-goal (lower is better).
        
        Args:
            batch: Dictionary containing:
                - states: [B, state_dim] (s_t)
                - next_states: [B, state_dim] (s_{t+1})
                - skip_states: [B, M, state_dim] (s_{t+h} for different h)
                - skip_steps: [B, M] (the h values as floats)
                - positive_goals: [B, goal_dim] (g^+)
                - positive_steps: [B] (d^+ = observed steps to positive goal)
                - unlabeled_goals: [B, K, goal_dim]
                - self_goals: [B, goal_dim] (s_t itself for identity constraint)
            grad_params: Parameters to compute gradients for.
        
        Returns:
            Tuple of (total_loss, info_dict)
        """
        states = batch['states']  # [B, state_dim] (s_t)
        next_states = batch['next_states']  # [B, state_dim] (s_{t+1})
        skip_states = batch['skip_states']  # [B, M, state_dim]
        skip_steps = batch['skip_steps']  # [B, M] (the h values)
        pos_goals = batch['positive_goals']  # [B, goal_dim]
        pos_steps = batch['positive_steps']  # [B] (d^+)
        unl_goals = batch['unlabeled_goals']  # [B, K, goal_dim]
        self_goals = batch.get('self_goals', states)  # [B, goal_dim] (defaults to states)
        
        # Expand positive steps to [B, 1] for broadcasting
        pos_distances = jnp.expand_dims(pos_steps, axis=1)  # [B, 1]
        
        B = states.shape[0]
        M = skip_states.shape[1] if len(skip_states.shape) > 2 else 0
        K = unl_goals.shape[1]
        
        # === CURRENT PREDICTIONS (trainable parameters) ===
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1]
        
        # Unlabeled predictions: R(s_t, g^(k)) for all k
        states_expanded = jnp.expand_dims(states, 1)  # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1))  # [B, K, state_dim]
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        pred_unl = self.network.select('value')(states_flat, unl_goals_flat, params=grad_params)
        pred_unl = pred_unl.reshape(B, K)  # [B, K]
        
        # === ANCHOR LOSSES (A4) ===
        # L_id: R(s, s) = 0
        pred_self = self.network.select('value')(states, self_goals, params=grad_params)  # [B, 1]
        loss_id = jnp.mean(jnp.square(pred_self))
        
        # L_edge: R(s_t, s_{t+1}) = 1
        pred_next = self.network.select('value')(states, next_states, params=grad_params)  # [B, 1]
        loss_edge = jnp.mean(jnp.square(pred_next - 1.0))
        
        # === MULTI-STEP BOUNDS (A2, A3) with Target Network ===
        if M > 0:
            # --- POSITIVE GOALS ---
            # Compute target values R_target(s_{t+h}, g^+) for all skip states
            skip_states_flat = skip_states.reshape(B * M, -1)
            pos_goals_expanded = jnp.expand_dims(pos_goals, 1)  # [B, 1, goal_dim]
            pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1))  # [B, M, goal_dim]
            pos_goals_flat = pos_goals_expanded.reshape(B * M, -1)
            
            target_pos_skip = self.network.select('target_value')(skip_states_flat, pos_goals_flat)
            target_pos_skip = target_pos_skip.reshape(B, M)  # [B, M]
            
            # Upper bound (A2): U_t(g) = min_h [h + R_target(s_{t+h}, g)]
            upper_bounds_pos = skip_steps + target_pos_skip  # [B, M]
            U_pos = jnp.min(upper_bounds_pos, axis=1, keepdims=True)  # [B, 1]
            
            # Lower bound (A3): L_t(g) = max_h [R_target(s_{t+h}, g) - h]
            lower_bounds_pos = target_pos_skip - skip_steps  # [B, M]
            L_pos = jnp.max(lower_bounds_pos, axis=1, keepdims=True)  # [B, 1]
            
            # L_UB: one-sided penalty when R(s_t, g^+) > U_t(g^+)
            loss_ub_pos = jnp.mean(jnp.square(jax.nn.relu(pred_pos - jax.lax.stop_gradient(U_pos))))
            
            # L_LB: one-sided penalty when R(s_t, g^+) < L_t(g^+)
            loss_lb_pos = jnp.mean(jnp.square(jax.nn.relu(jax.lax.stop_gradient(L_pos) - pred_pos)))
            
            # --- UNLABELED GOALS ---
            # Expand skip states for all unlabeled goals: [B, M, K, state_dim]
            skip_states_exp = jnp.expand_dims(skip_states, 2)  # [B, M, 1, state_dim]
            skip_states_exp = jnp.tile(skip_states_exp, (1, 1, K, 1))  # [B, M, K, state_dim]
            skip_states_exp_flat = skip_states_exp.reshape(B * M * K, -1)
            
            unl_goals_exp = jnp.expand_dims(unl_goals, 1)  # [B, 1, K, goal_dim]
            unl_goals_exp = jnp.tile(unl_goals_exp, (1, M, 1, 1))  # [B, M, K, goal_dim]
            unl_goals_exp_flat = unl_goals_exp.reshape(B * M * K, -1)
            
            target_unl_skip = self.network.select('target_value')(skip_states_exp_flat, unl_goals_exp_flat)
            target_unl_skip = target_unl_skip.reshape(B, M, K)  # [B, M, K]
            
            # Expand skip_steps for unlabeled: [B, M, K]
            skip_steps_exp = jnp.expand_dims(skip_steps, 2)  # [B, M, 1]
            skip_steps_exp = jnp.tile(skip_steps_exp, (1, 1, K))  # [B, M, K]
            
            # Upper and lower bounds for unlabeled
            upper_bounds_unl = skip_steps_exp + target_unl_skip  # [B, M, K]
            U_unl = jnp.min(upper_bounds_unl, axis=1)  # [B, K]
            
            lower_bounds_unl = target_unl_skip - skip_steps_exp  # [B, M, K]
            L_unl = jnp.max(lower_bounds_unl, axis=1)  # [B, K]
            
            loss_ub_unl = jnp.mean(jnp.square(jax.nn.relu(pred_unl - jax.lax.stop_gradient(U_unl))))
            loss_lb_unl = jnp.mean(jnp.square(jax.nn.relu(jax.lax.stop_gradient(L_unl) - pred_unl)))
            
            loss_ub = 0.5 * (loss_ub_pos + loss_ub_unl)
            loss_lb = 0.5 * (loss_lb_pos + loss_lb_unl)
        else:
            loss_ub = 0.0
            loss_lb = 0.0
            U_pos = jnp.zeros_like(pred_pos)
            L_pos = jnp.zeros_like(pred_pos)
            U_unl = jnp.zeros_like(pred_unl)
            L_unl = jnp.zeros_like(pred_unl)
        
        # === HINDSIGHT POSITIVE BOUND (A6) ===
        # R(s_t, g^+) ≤ d^+ (one-sided penalty)
        loss_pos = jnp.mean(jnp.square(jax.nn.relu(pred_pos - pos_distances)))
        
        # === PU RANKING LOSS (A5) ===
        # For cost (lower is better): R(s_t, g^+) + m ≤ Agg[R(s_t, g^(k))]
        # Rearranged: Agg[R(s_t, g^(k))] - R(s_t, g^+) - m ≥ 0
        agg_unl = jnp.mean(pred_unl, axis=1, keepdims=True)  # [B, 1] (mean aggregation)
        rank_logits = agg_unl - pred_pos - self.config['rank_margin']
        loss_rank = jnp.mean(-agg_unl)
        #loss_rank = jnp.mean(jax.nn.softplus(-rank_logits))
        #loss_rank = (jax.nn.softplus(self.config.get('max_horizon', 100.0)- agg_unl)).mean()
        

        lam = self.network.select('lam')(params=grad_params)
        lam_loss = lam * (0.05 - jax.lax.stop_gradient(loss_lb))


        # === TOTAL LOSS ===
        total_loss = (
            self.config.get('lambda_ub', 1.0) * loss_ub 
            + self.config.get('lambda_lb', 1.0) * loss_lb * jax.lax.stop_gradient(lam)
            #+ self.config.get('lambda_pos', 1.0) * loss_pos 
            + self.config.get('lambda_rank', 1.0) * loss_rank 
            #+ self.config.get('lambda_id', 1.0) * loss_id 
            #+ loss_edge * jax.lax.stop_gradient(lam)
            + lam_loss
        )
        
        info = {
            'loss_total': total_loss,
            'loss_ub': loss_ub,
            'loss_lb': loss_lb,
            'loss_id': loss_id,
            'loss_edge': loss_edge,
            'loss_pos': loss_pos,
            'loss_rank': loss_rank,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'pred_self_mean': jnp.mean(pred_self),
            'pred_next_mean': jnp.mean(pred_next),
            'pos_steps_mean': jnp.mean(pos_steps),
        }
        
        if M > 0:
            info.update({
                'U_pos_mean': jnp.mean(U_pos),
                'L_pos_mean': jnp.mean(L_pos),
                'U_unl_mean': jnp.mean(U_unl),
                'L_unl_mean': jnp.mean(L_unl),
            })
        
        return total_loss, info
    
    def reachability_loss(self, batch, grad_params):
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
        """Predict step-cost between observations and goals.
        
        Args:
            observations: [B, obs_dim] or [obs_dim]
            goals: [B, goal_dim] or [goal_dim]
        
        Returns:
            Step costs R(s,g) in [0, H_max], shape [B, 1] or [1]
            Lower values indicate closer/more reachable goals.
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

        # Define step-cost value network (outputs in [0, H_max])
        #'''
        value_def = StepsValue(
            hidden_dims=config['value_hidden_dims'],
            h_max=config.get('max_horizon', 100.0),
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )
        '''
        
        value_def = GCMRNValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            encoder=encoders.get('value'),
        )
        
        '''
        '''
        value_def = GCIQEValue(
                dim_per_component=8,
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                encoder=encoders.get('value'),
                h_max=config.get('max_horizon', 100.0),
            )
        '''
        # Define the dual lambda variable.
        lam_def = LogParam()

        # Create network with both value and target_value
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            lam=(lam_def, ()),
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
            agent_name='rws',
            lr=3e-4,
            batch_size=1024,
            value_hidden_dims=(256, 256, 256),
            layer_norm=True,
            tau=0.995,  # EMA rate for target network
            latent_dim=64,
            
            # Loss weights (Version A)
            lambda_ub=1.0,      # Upper bound loss weight
            lambda_lb=1.0,      # Lower bound loss weight
            lambda_id=1.0,      # Identity constraint weight (R(s,s)=0)
            lambda_edge=1.0,    # Edge cost constraint weight (R(s_t,s_{t+1})=1)
            lambda_pos=1.0,     # Positive bound weight (R(s,g^+) ≤ d^+)
            lambda_rank=.01,    # Ranking loss weight
            rank_margin=0.0,    # Margin m for ranking (in steps)
            
            # Network output range
            max_horizon=100.0,  # H_max: maximum output value for R(s,g)
            
            # Dataset hyperparameters
            dataset_class='ReachabilityDataset',
            num_goals_per_state=4,      # K: number of unlabeled goals
            num_skip_states=3,          # M: number of skip states
            max_skip_horizon=None,      # Maximum h value (None = trajectory end)
            
            # GCDataset config (inherited by ReachabilityDataset)
            discount=0.99,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,       # Sample positive goals from trajectory
            value_p_randomgoal=0.0,
            value_geom_sample=False,    # Uniform sampling for future goals
            actor_p_curgoal=0.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config