import copy
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
#from utils.networks import RWSValue



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

class RWSAgent(flax.struct.PyTreeNode):
    """Reachability estimator agent using the loss from rws_test.py.
    
    This agent trains a value network to predict reachability between states and goals
    using PU-RANK loss and multi-step consistency loss with a target network.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def reachability_loss_(self, batch, grad_params):
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

    def reachability_loss_PUNCE(self, batch, grad_params):
        """Compute the combined reachability loss (PU-NCE + Consistency)."""
        states = batch['states']          # [B, state_dim]
        skip_states = batch['skip_states']# [B, M, state_dim]
        pos_goals = batch['positive_goals']      # [B, goal_dim]
        unl_goals = batch['unlabeled_goals']     # [B, K, goal_dim]

        B = states.shape[0]
        M = skip_states.shape[1]
        K = unl_goals.shape[1]

        # === 0. Forward passes for positives & unlabeled ===
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1]

        states_expanded = jnp.expand_dims(states, 1)           # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1)) # [B, K, state_dim]
        unl_goals_flat = unl_goals.reshape(B * K, -1)          # [B*K, goal_dim]
        states_flat = states_expanded.reshape(B * K, -1)       # [B*K, state_dim]
        pred_unl = self.network.select('value')(states_flat, unl_goals_flat, params=grad_params).reshape(B, K)  # [B, K]

        # === 1. Target multi-step max for unlabeled (moved up so PU-NCE can use it) ===
        # Expand skip states for all unlabeled: [B, M, K, state_dim]
        skip_states_expanded_unl = jnp.expand_dims(skip_states, 2)         # [B, M, 1, state_dim]
        skip_states_expanded_unl = jnp.tile(skip_states_expanded_unl, (1, 1, K, 1)).reshape(B * M * K, -1)

        # Expand unlabeled goals across M: [B, M, K, goal_dim]
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)                 # [B, 1, K, goal_dim]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1)).reshape(B * M * K, -1)

        target_unl_all = self.network.select('target_value')(skip_states_expanded_unl, unl_goals_expanded)
        target_unl_all = target_unl_all.reshape(B, M, K)                    # [B, M, K]
        target_unl_max = jnp.max(target_unl_all, axis=1)                    # [B, K]  (stop-grad will be applied below)

        # === 2. PU-NCE (contrastive with soft pseudo-negative weights) ===
        tau  = self.config.get('pu_tau', 0.3)
        beta = self.config.get('pu_beta', 3.0)
        eps  = 1e-8

        # q_unl ≈ probability an unlabeled is actually positive (no grad)
        q_unl = jax.nn.sigmoid(beta * jax.lax.stop_gradient(target_unl_max))  # [B, K]
        # weights favor likely negatives
        w_unl = 1.0 - q_unl                                                   # [B, K]
        w_unl = w_unl / (jnp.sum(w_unl, axis=1, keepdims=True) + eps)         # normalize over K

        pos = pred_pos.squeeze(-1)                                            # [B]
        num = jnp.exp(pos / tau)                                              # [B]
        den = num + jnp.sum(w_unl * jnp.exp(pred_unl / tau), axis=1)          # [B]
        rank_loss = -jnp.mean(jnp.log(num / (den + eps)))                     # PU-NCE

        # === 3. MULTI-STEP CONSISTENCY LOSS WITH MAX-POOLING ===
        # Positives: max_i r_target(s_{t+hi}, g^+)
        skip_states_flat = skip_states.reshape(B * M, -1)                     # [B*M, state_dim]
        pos_goals_expanded = jnp.expand_dims(pos_goals, 1)                    # [B, 1, goal_dim]
        pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1)).reshape(B * M, -1)

        target_pos_all = self.network.select('target_value')(skip_states_flat, pos_goals_expanded)
        target_pos_all = target_pos_all.reshape(B, M)                          # [B, M]
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)        # [B, 1]

        cons_pos = jnp.mean(jnp.square(pred_pos - jax.lax.stop_gradient(target_pos_max)))

        # Unlabeled consistency: reuse target_unl_max computed above
        cons_unl = jnp.mean(jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl))

        consistency_loss = 0.5 * (cons_pos + cons_unl)

        # === 4. TOTAL ===
        total_loss = rank_loss + self.config['lambda_cons'] * consistency_loss

        info = {
            'loss_total': total_loss,
            'loss_rank': rank_loss,                 # now PU-NCE
            'loss_cons': consistency_loss,
            'cons_pos': cons_pos,
            'cons_unl': cons_unl,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'target_pos_max_mean': jnp.mean(target_pos_max),
            'target_unl_max_mean': jnp.mean(target_unl_max),
            # Optional diagnostics for PU-NCE:
            'pu_q_unl_mean': jnp.mean(q_unl),
        }

        return total_loss, info

    def reachability_loss_adptiveHinge(self, batch, grad_params):
        """Compute the combined reachability loss (PU-RANK + Consistency)."""
        states = batch['states']              # [B, state_dim]
        skip_states = batch['skip_states']    # [B, M, state_dim]
        pos_goals = batch['positive_goals']   # [B, goal_dim]
        unl_goals = batch['unlabeled_goals']  # [B, K, goal_dim]
        
        B = states.shape[0]
        M = skip_states.shape[1]
        K = unl_goals.shape[1]

        # === 1. Scores for positives & unlabeled ===
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B, 1]

        # states -> [B,K,state_dim], flatten with unl_goals for batch scoring
        states_expanded = jnp.expand_dims(states, 1)
        states_expanded = jnp.tile(states_expanded, (1, K, 1))
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        pred_unl = self.network.select('value')(states_flat, unl_goals_flat, params=grad_params).reshape(B, K)  # [B,K]

        # === 2. MULTI-STEP CONSISTENCY LOSS WITH MAX-POOLING (unchanged) ===
        # Positives
        skip_states_flat = skip_states.reshape(B * M, -1)
        pos_goals_expanded = jnp.expand_dims(pos_goals, 1)
        pos_goals_expanded = jnp.tile(pos_goals_expanded, (1, M, 1)).reshape(B * M, -1)
        target_pos_all = self.network.select('target_value')(skip_states_flat, pos_goals_expanded).reshape(B, M)  # [B,M]
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B,1]
        cons_pos = jnp.mean(jnp.square(pred_pos - jax.lax.stop_gradient(target_pos_max)))

        # Unlabeled
        skip_states_expanded_unl = jnp.expand_dims(skip_states, 2)              # [B,M,1,D]
        skip_states_expanded_unl = jnp.tile(skip_states_expanded_unl, (1, 1, K, 1)).reshape(B * M * K, -1)
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)                       # [B,1,K,G]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1)).reshape(B * M * K, -1)
        target_unl_all = self.network.select('target_value')(skip_states_expanded_unl, unl_goals_expanded).reshape(B, M, K)
        target_unl_max = jnp.max(target_unl_all, axis=1)                         # [B,K]
        cons_unl = jnp.mean(jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl))
        consistency_loss = 0.5 * (cons_pos + cons_unl)

        # === 3. NEW: Adaptive-margin PU hinge (replaces previous rank loss) ===
        # m_{ik} = m0 * (1 - stopgrad(target_unl_max))^gamma
        # logits_{ik} = (pred_unl_{ik} - pred_pos_i) + m_{ik}
        # rank_loss = mean softplus(logits)
        m0 = self.config.get('pu_margin', self.config.get('rank_margin', 0.))  # fall back to your old margin if present
        gamma = self.config.get('pu_gamma', 1)

        m_ik = m0 * jnp.power(1.0 - jax.lax.stop_gradient(target_unl_max), gamma)  # [B,K]
        logits = (pred_unl - pred_pos) + m_ik                                      # broadcast pred_pos [B,1] -> [B,K]
        rank_loss = jnp.mean(jax.nn.softplus(logits))

        # === 4. TOTAL ===
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
            # optional diagnostics:
            'pu_margin': jnp.mean(m_ik),
            'pu_gamma': gamma,
        }
        return total_loss, info

    def reachability_loss_selfpaced(self, batch, grad_params):
        """PU-InfoNCE rank + multi-step consistency with self-paced negatives from target max-pool."""
        states = batch['states']              # [B, state_dim]
        skip_states = batch['skip_states']    # [B, M, state_dim]
        pos_goals = batch['positive_goals']   # [B, goal_dim]
        unl_goals = batch['unlabeled_goals']  # [B, K, goal_dim]

        B = states.shape[0]
        M = skip_states.shape[1]
        K = unl_goals.shape[1]

        eps = 1e-8
        margin = self.config.get('rank_margin', 0.2)
        tau    = self.config.get('rank_tau', 1.0)
        Tsp    = self.config.get('self_paced_T', 1.0)

        # -------------------------
        # 1) Forward passes we need
        # -------------------------
        # r(s, g+)
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)  # [B,1]

        # r(s, g_unl)
        states_expanded = jnp.tile(states[:, None, :], (1, K, 1))       # [B,K,Ds]
        pred_unl = self.network.select('value')(
            states_expanded.reshape(B * K, -1),
            unl_goals.reshape(B * K, -1),
            params=grad_params
        ).reshape(B, K)  # [B,K]

        # target max-pooled for positives (for consistency and diagnostics)
        skip_states_flat = skip_states.reshape(B * M, -1)  # [B*M, Ds]
        pos_goals_expanded = jnp.tile(pos_goals[:, None, :], (1, M, 1)).reshape(B * M, -1)  # [B*M,Dg]
        target_pos_all = self.network.select('target_value')(skip_states_flat, pos_goals_expanded).reshape(B, M)  # [B,M]
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B,1]

        # target max-pooled for unlabeled (our self-paced proxy)
        # [B,M,K,Ds] and [B,M,K,Dg] -> [B*M*K, Ds/Dg]
        skip_states_expanded_unl = jnp.tile(skip_states[:, :, None, :], (1, 1, K, 1)).reshape(B * M * K, -1)
        unl_goals_expanded = jnp.tile(unl_goals[:, None, :, :], (1, M, 1, 1)).reshape(B * M * K, -1)
        target_unl_all = self.network.select('target_value')(skip_states_expanded_unl, unl_goals_expanded).reshape(B, M, K)  # [B,M,K]
        target_unl_max = jnp.max(target_unl_all, axis=1)  # [B,K]

        # ------------------------------------------------------
        # 2) PU-InfoNCE with self-paced negative weights (NEW)
        # ------------------------------------------------------
        # row-wise threshold (median) on stop-grad target scores
        tau0 = jnp.median(target_unl_max, axis=1, keepdims=True)  # [B,1]
        w_neg = jax.nn.sigmoid((tau0 - jax.lax.stop_gradient(target_unl_max)) / Tsp)  # [B,K]
        w_neg = w_neg / (jnp.sum(w_neg, axis=1, keepdims=True) + eps)                 # [B,K], normalized

        # logits with margin/temperature
        pos_logits = (pred_pos - margin) / tau            # [B,1]
        unl_logits = pred_unl / tau                       # [B,K]

        # denom = log( exp(pos) + sum_i w_i * exp(unl_i) )
        from jax.scipy.special import logsumexp
        log_weighted_unl = jnp.log(w_neg + eps) + unl_logits                 # [B,K]
        denom = jnp.logaddexp(pos_logits, logsumexp(log_weighted_unl, axis=1, keepdims=True))  # [B,1]
        rank_loss = jnp.mean(denom - pos_logits)  # -log softmax prob of the positive

        # ----------------------------------------------
        # 3) Multi-step consistency (unchanged semantics)
        # ----------------------------------------------
        # Positives: regression to target max
        cons_pos = jnp.mean(jnp.square(pred_pos - jax.lax.stop_gradient(target_pos_max)))

        # Unlabeled: only pull up when target > current
        cons_unl = jnp.mean(jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl))

        consistency_loss = 0.5 * (cons_pos + cons_unl)

        # ---------------
        # 4) Total + info
        # ---------------
        total_loss = rank_loss #+ self.config['lambda_cons'] * consistency_loss

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
            # diagnostics for the self-paced mechanism:
            'w_neg_mean': jnp.mean(w_neg),
            'w_neg_frac_small': jnp.mean((w_neg < 0.1).astype(jnp.float32)),
            'tau0_median_mean': jnp.mean(tau0),
        }

        return total_loss, info

    def reachability_loss_baclup(self, batch, grad_params):
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

        tail = 0.3
        h_max = self.config.get('max_horizon', 1100.0)
        gamma        = float(tail) ** (1.0 / max(h_max, 1.0))

        margin       = self.config.get('rank_margin', 0.001)
        lambda_mono  = self.config.get('lambda_mono', self.config.get('lambda_cons', 1.0))  # fallback to old key

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

        # === TD-style monotone inequality against target net (no Q): f(s, g) >= gamma * max_m f_bar(s_mid^m, g) ===
        # 1) Positives: compute target on skip states for each (s_i, g^+_i), then max over M
        skip_flat = skip_states.reshape(B*M, -1)           # [B*M, state_dim]
        pos_goals_exp = jnp.repeat(jnp.expand_dims(pos_goals, 1), M, axis=1).reshape(B*M, -1)  # [B*M, goal_dim]

        target_pos_all = self.network.select('target_value')(skip_flat, pos_goals_exp)  # [B*M, 1] or [B*M]
        target_pos_all = jnp.squeeze(target_pos_all, axis=-1).reshape(B, M)             # [B, M]
        target_pos_max = jnp.max(target_pos_all, axis=1)                                 # [B]

        # Monotone penalty for positives: relu(gamma * target_max - current)
        mono_pos = jax.nn.relu(jax.lax.stop_gradient(gamma * target_pos_max) - pred_pos) # [B]
        mono_pos = jnp.mean(mono_pos)

        # 2) Unlabeled: compute target on skip states for each (s_i, g~_{i,j}), then max over M, for all j
        # Expand (s_mid, g_unl) pairs -> [B, M, K, ...] flattened once
        skip_exp_unl = jnp.expand_dims(skip_states, 2)                 # [B, M, 1, state_dim]
        skip_exp_unl = jnp.broadcast_to(skip_exp_unl, (B, M, K, skip_states.shape[-1]))  # [B, M, K, state_dim]
        skip_exp_unl = skip_exp_unl.reshape(B*M*K, -1)                  # [B*M*K, state_dim]

        unl_goals_exp = jnp.expand_dims(unl_goals, 1)                   # [B, 1, K, goal_dim]
        unl_goals_exp = jnp.broadcast_to(unl_goals_exp, (B, M, K, unl_goals.shape[-1]))
        unl_goals_exp = unl_goals_exp.reshape(B*M*K, -1)                # [B*M*K, goal_dim]

        target_unl_all = self.network.select('target_value')(skip_exp_unl, unl_goals_exp)  # [B*M*K, 1] or [B*M*K]
        target_unl_all = jnp.squeeze(target_unl_all, axis=-1).reshape(B, M, K)             # [B, M, K]
        target_unl_max = jnp.max(target_unl_all, axis=1)                                    # [B, K]

        # Monotone penalty for unlabeled (pointwise): relu(gamma * target_max - current)
        mono_unl = jax.nn.relu(jax.lax.stop_gradient(gamma * target_unl_max) - pred_unl)    # [B, K]
        mono_unl = jnp.mean(mono_unl)

        mono_loss = 0.5 * (mono_pos + mono_unl)

        # === TOTAL LOSS ===
        total_loss = rank_loss + lambda_mono * mono_loss

        info = {
            'loss_total': total_loss,
            'loss_rank': rank_loss,
            'loss_mono': mono_loss,
            'mono_pos': mono_pos,
            'mono_unl': mono_unl,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'tgt_pos_max_mean': jnp.mean(target_pos_max),
            'tgt_unl_max_mean': jnp.mean(target_unl_max),
        }
        return total_loss, info
   

    def reachability_loss_(self, batch, grad_params):
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
        
        tail = 0.3
        h_max = self.config.get('max_horizon', 2000.0)
        gamma        = float(tail) ** (1.0 / max(h_max, 1.0))

        margin       = self.config.get('rank_margin', 0.001)
        lambda_mono  = self.config.get('lambda_mono', self.config.get('lambda_cons', 1.0))  # fallback to old key
        
        # Warmup configuration
        warmup_steps = self.config.get('mono_warmup_steps', 24000)  # 0 = no warmup
        step = batch.get('train_step', batch.get('global_step', batch.get('step', 0)))
        
        # Compute warmup coefficient (0 -> 1 over warmup_steps)
        if warmup_steps > 0:
            warmup_coeff = jnp.minimum(1.0, float(step) / float(warmup_steps))
        else:
            warmup_coeff = 1.0

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

        # === TD-style monotone inequality against target net (no Q): f(s, g) >= gamma * max_m f_bar(s_mid^m, g) ===
        # 1) Positives: compute target on skip states for each (s_i, g^+_i), then max over M
        skip_flat = skip_states.reshape(B*M, -1)           # [B*M, state_dim]
        pos_goals_exp = jnp.repeat(jnp.expand_dims(pos_goals, 1), M, axis=1).reshape(B*M, -1)  # [B*M, goal_dim]

        target_pos_all = self.network.select('target_value')(skip_flat, pos_goals_exp)  # [B*M, 1] or [B*M]
        target_pos_all = jnp.squeeze(target_pos_all, axis=-1).reshape(B, M)             # [B, M]
        target_pos_max = jnp.max(target_pos_all, axis=1)                                 # [B]

        # Monotone penalty for positives: relu(gamma * target_max - current)
        mono_pos = jax.nn.relu(jax.lax.stop_gradient(gamma * target_pos_max) - pred_pos) # [B]
        mono_pos = jnp.mean(mono_pos)

        # 2) Unlabeled: compute target on skip states for each (s_i, g~_{i,j}), then max over M, for all j
        # Expand (s_mid, g_unl) pairs -> [B, M, K, ...] flattened once
        skip_exp_unl = jnp.expand_dims(skip_states, 2)                 # [B, M, 1, state_dim]
        skip_exp_unl = jnp.broadcast_to(skip_exp_unl, (B, M, K, skip_states.shape[-1]))  # [B, M, K, state_dim]
        skip_exp_unl = skip_exp_unl.reshape(B*M*K, -1)                  # [B*M*K, state_dim]

        unl_goals_exp = jnp.expand_dims(unl_goals, 1)                   # [B, 1, K, goal_dim]
        unl_goals_exp = jnp.broadcast_to(unl_goals_exp, (B, M, K, unl_goals.shape[-1]))
        unl_goals_exp = unl_goals_exp.reshape(B*M*K, -1)                # [B*M*K, goal_dim]

        target_unl_all = self.network.select('target_value')(skip_exp_unl, unl_goals_exp)  # [B*M*K, 1] or [B*M*K]
        target_unl_all = jnp.squeeze(target_unl_all, axis=-1).reshape(B, M, K)             # [B, M, K]
        target_unl_max = jnp.max(target_unl_all, axis=1)                                    # [B, K]

        # Monotone penalty for unlabeled (pointwise): relu(gamma * target_max - current)
        mono_unl = jax.nn.relu(jax.lax.stop_gradient(gamma * target_unl_max) - pred_unl)    # [B, K]
        mono_unl = jnp.mean(mono_unl)

        mono_loss = 0.5 * (mono_pos + mono_unl)

        # === TOTAL LOSS (with warmup) ===
        total_loss = rank_loss + lambda_mono * warmup_coeff * mono_loss

        info = {
            'loss_total': total_loss,
            'loss_rank': rank_loss,
            'loss_mono': mono_loss,
            'mono_pos': mono_pos,
            'mono_unl': mono_unl,
            'warmup_coeff': warmup_coeff,
            'pred_pos_mean': jnp.mean(pred_pos),
            'pred_unl_mean': jnp.mean(pred_unl),
            'tgt_pos_max_mean': jnp.mean(target_pos_max),
            'tgt_unl_max_mean': jnp.mean(target_unl_max),
        }
        return total_loss, info
    


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

        pointmaze-medium-navigate-v0 # 0.8
        pointmaze-large-navigate-v0 # 0.8
        pointmaze-giant-navigate-v0 # 0.8
        pointmaze-teleport-navigate-v0 # 0.99
        pointmaze-medium-stitch-v0 # 0.99
        pointmaze-large-stitch-v0 # 0.9999
        pointmaze-giant-stitch-v0 # 1
        pointmaze-teleport-stitch-v0 # 0.9999
        antmaze-medium-navigate-v0 # 0.9
        antmaze-large-navigate-v0 # 0.9
        antmaze-giant-navigate-v0 # 0.9
        antmaze-teleport-navigate-v0 # 0.99
        antmaze-medium-stitch-v0 # 0.999
        antmaze-large-stitch-v0 # 0.999
        antmaze-giant-stitch-v0 # 0.999
        antmaze-teleport-stitch-v0 # 0.999
        antmaze-medium-explore-v0 # 0.9995
        antmaze-large-explore-v0 # 0.9995
        antmaze-teleport-explore-v0 # 0.9995
        humanoidmaze-medium-navigate-v0 # gamma 0.9
        humanoidmaze-large-navigate-v0 # gamma 0.9
        humanoidmaze-giant-navigate-v0 # gamma 0.9
        humanoidmaze-medium-stitch-v0 # 0.999
        humanoidmaze-large-stitch-v0 # 0.999
        humanoidmaze-giant-stitch-v0 # gamma 0.9995
        antsoccer-arena-navigate-v0
        antsoccer-medium-navigate-v0
        antsoccer-arena-stitch-v0
        antsoccer-medium-stitch-v0

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

        tail = 0.3
        h_max = self.config.get('max_horizon', 1100.0)
        gamma        = float(tail) ** (1.0 / max(1500, 1.0))
        gamma = 0.99 # point 0.99
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
        gamma_weights_unl = gamma_weights.reshape(1, M, 1)  # [1, M, 1] for broadcasting over K
        target_unl_all_weighted = target_unl_all * gamma_weights_unl  # [B, M, K]
        target_unl_max = jnp.max(target_unl_all_weighted, axis=1)      # [B, K]

        # Monotone penalty for unlabeled (pointwise): relu(target_max - current)
        mono_unl = jax.nn.relu(jax.lax.stop_gradient(target_unl_max) - pred_unl)    # [B, K]
        mono_unl = jnp.mean(mono_unl)

        mono_loss = 0.5 * (mono_pos + mono_unl)

        # === TOTAL LOSS ===
        total_loss = rank_loss + lambda_mono * mono_loss

        info = {
            'loss_total': total_loss,
            'loss_rank': rank_loss,
            'loss_mono': mono_loss,
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
    def create_origin(
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