"""
Goal-conditioned Expectile Steps Agent with Negative Horizon Prediction
Predicts -H(s,g) ∈ [-H_max, 0] where higher (less negative) is better
Uses expectile regression to learn upper bound of negative horizons
"""

import copy
from typing import Any, Sequence, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class StepsValue(nn.Module):
    """Value network that predicts negative cost-to-go (negative steps).
    
    Outputs -H(s,g) ∈ [-H_max, 0] representing negative predicted minimum steps from s to g.
    Higher is better (less negative = fewer steps = better).
    
    Uses -H_max * sigmoid(z) to ensure output is in [-H_max, 0].
    """
    hidden_dims: Sequence[int]
    h_max: float = 100.0
    layer_norm: bool = True
    ensemble: bool = False
    gc_encoder: Optional[nn.Module] = None
    
    @nn.compact
    def __call__(self, observations, goals):
        """Predict negative step costs from observations to goals.
        
        Args:
            observations: [B, obs_dim]
            goals: [B, goal_dim]
        
        Returns:
            If ensemble=True: Tuple of (v1, v2), each shape [B, 1]
            If ensemble=False: Single value, shape [B, 1]
            Values in [-H_max, 0], higher = better (closer to goal)
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
        
        if self.ensemble:
            # Two separate output heads for ensemble
            logits1 = nn.Dense(1, name='fc_out1')(x)
            logits2 = nn.Dense(1, name='fc_out2')(x)
            
            # Map to [-H_max, 0] using negative scaled sigmoid
            v1 = -self.h_max * nn.sigmoid(logits1)
            v2 = -self.h_max * nn.sigmoid(logits2)
            
            return v1, v2
        else:
            # Single output head
            logits = nn.Dense(1, name='fc_out')(x)
            negative_steps = -self.h_max * nn.sigmoid(logits)
            return negative_steps


class ExpectileStepsAgent(flax.struct.PyTreeNode):
    """Goal-conditioned expectile steps prediction agent with negative horizon prediction.
    
    Follows GCIVL structure but predicts -H(s,g) ∈ [-H_max, 0] instead of values.
    Higher (less negative) values are better (fewer steps to goal).
    Expectile τ > 0.5 learns optimistic (upper bound) estimates.
    
    No actor network. Uses the 'policy' batch from ReachabilityGCDataset which has
    GCIVL-style keys (observations, next_observations, value_goals, rewards, masks).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss (identical to GCIVL)."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        """Compute the expectile negative-steps loss.
        
        Follows GCIVL's value_loss structure but adapted for negative horizon prediction:
        - Predicts -H(s,g) ∈ [-H_max, 0] where higher is better
        - Target: -step_increment - mask * next_V
        - Expectile τ > 0.5 learns upper bound (less negative = shorter paths)
        
        The 'rewards' in batch indicate goal achievement: 0 if goal reached, -1 otherwise.
        The 'masks' are: 0 if goal reached, 1 otherwise.
        """
        # Get negative step predictions for next states using target network
        (next_v1_t, next_v2_t) = self.network.select('target_value')(
            batch['next_observations'], batch['value_goals']
        )
        # Use max for negative values (less negative = better)
        next_v_t = jnp.maximum(next_v1_t, next_v2_t)
        
        # Compute negative step target (following GCIVL structure)
        # step_increment = 1 when not at goal, 0 when at goal
        # We negate to get negative horizon: -1 when not at goal, 0 when at goal
        step_increment = batch['masks']  # 1 if not at goal, 0 if at goal
        negative_step_increment = -step_increment  # -1 if not at goal, 0 if at goal
        
        # Bellman-style target for negative horizons
        # If not at goal: target = -1 + mask * next_v = -1 + next_v (accumulate negative steps)
        # If at goal: target = 0 + 0 * next_v = 0 (no more steps needed)
        target = negative_step_increment + batch['masks'] * next_v_t

        # Get current predictions using target network (for advantage)
        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], batch['value_goals']
        )
        v_t = (v1_t + v2_t) / 2
        adv = target - v_t
        
        # Compute targets for each ensemble member
        target1 = negative_step_increment + batch['masks'] * next_v1_t
        target2 = negative_step_increment + batch['masks'] * next_v2_t
        
        # Get current predictions with gradients
        (v1, v2) = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        v = (v1 + v2) / 2

        # Compute expectile loss (same structure as GCIVL)
        value_loss1 = self.expectile_loss(adv, target1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, target2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'neg_steps_mean': v.mean(),
            'neg_steps_max': v.max(),  # Less negative = better
            'neg_steps_min': v.min(),  # More negative = worse
            'target_mean': target.mean(),
            'advantage_mean': adv.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (only value loss, no actor)."""
        info = {}
        
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        return value_loss, info

    def target_update(self, network, module_name):
        """Update the target network (same as GCIVL)."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def predict_negative_steps(self, observations, goals):
        """Predict negative steps-to-go from observations to goals.
        
        Returns:
            Negative steps in [-h_max, 0], where higher (less negative) is better.
        """
        v1, v2 = self.network.select('value')(observations, goals)
        neg_steps = (v1 + v2) / 2
        return neg_steps

    @jax.jit
    def predict_steps(self, observations, goals):
        """Predict positive steps-to-go from observations to goals.
        
        Returns:
            Positive steps in [0, h_max], where lower is better.
        """
        neg_steps = self.predict_negative_steps(observations, goals)
        # Convert negative to positive: -(-H) = H
        return -neg_steps

    @jax.jit
    def predict_reachability(self, observations, goals):
        """Convert steps prediction to reachability score (for visualization).
        
        Reachability = 1 - steps / h_max (clamped to [0, 1])
        """
        steps = self.predict_steps(observations, goals)
        reachability = 1.0 - steps / self.config['h_max']
        return jnp.clip(reachability, 0.0, 1.0)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,  # Not used but kept for interface compatibility
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions (not used, for compatibility).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations

        # Define encoders (same as GCIVL)
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())

        # Define StepsValue networks (predicts negative horizons)
        value_def = StepsValue(
            hidden_dims=config['value_hidden_dims'],
            h_max=config['h_max'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('value'),
        )
       
        # Only value networks (no actor)
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

        # Initialize target network (same as GCIVL)
        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Get default configuration for ExpectileStepsAgent."""
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters (following GCIVL structure)
            agent_name='expectile_steps',
            lr=3e-4,
            batch_size=1024,
            value_hidden_dims=(256, 256, 256),
            layer_norm=True,
            discount=0.99,  # Not directly used in step prediction, but kept for compatibility
            tau=0.005,  # Target network update rate
            expectile=0.7,  # Expectile parameter (>0.5 = optimistic upper bound)
            h_max=100.0,  # Maximum horizon for normalization
            
            # Encoder settings
            encoder=None,  # Visual encoder name (None, 'impala_small', etc.)
            frame_stack=None,  # Number of frames to stack
            
            # Dataset settings (for ReachabilityGCDataset)
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=0.,
            value_p_randomgoal=1.,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,  # Use -1 reward for non-goal states
            p_aug=None,
        )
    )
    return config