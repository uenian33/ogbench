import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import RWSValue


class RWSAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent - Critic learning only.

    This implementation focuses only on value and critic learning without the actor.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def reach_loss(self, batch, grad_params):
        states = batch["states"]  # [B, state_dim]
        skip_states = batch["skip_states"]  # [B, M, state_dim]
        pos_goals = batch["positive_goals"]  # [B, goal_dim]
        unl_goals = batch["unlabeled_goals"]  # [B, K, goal_dim]
        self_goals = batch["self_goals"]  # [B, goal_dim]
        rank_margin = -0.05
        lambda_cons = 1

        B, M, state_dim = skip_states.shape
        K = unl_goals.shape[1]
        
        # === 1. PU-RANK LOSS ===
        pred_pos = self.network.select('value')(states, pos_goals, params=grad_params)
        
        # Compute predictions for all K unlabeled goals
        states_expanded = jnp.expand_dims(states, 1)  # [B, 1, state_dim]
        states_expanded = jnp.tile(states_expanded, (1, K, 1))  # [B, K, state_dim]
        unl_goals_flat = unl_goals.reshape(B * K, -1)
        states_flat = states_expanded.reshape(B * K, -1)
        #pred_unl = state.apply_fn(params, states_flat, unl_goals_flat).reshape(B, K)
        pred_unl = self.network.select('value')(states_flat, unl_goals_flat, params=grad_params).reshape(B, K)
        
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
        #target_pos_all = state.apply_fn(state.target_params, skip_states_flat, pos_goals_expanded)
        target_pos_all = self.network.select('target_value')(skip_states_flat, pos_goals_expanded)
        target_pos_all = target_pos_all.reshape(B, M)
        target_pos_max = jnp.max(target_pos_all, axis=1, keepdims=True)  # [B, 1]
        
        #cons_pos = jnp.mean(jax.nn.relu(target_pos_max - pred_pos))
        cons_pos = -jnp.mean(target_pos_all)  - jnp.mean(pred_pos)

        # For unlabeled: max_i r̄(s_{t+hi}, g_unl)
        skip_states_expanded_unl = jnp.expand_dims(skip_states, 2)  # [B, M, 1, state_dim]
        skip_states_expanded_unl = jnp.tile(skip_states_expanded_unl, (1, 1, K, 1))  # [B, M, K, state_dim]
        skip_states_expanded_unl = skip_states_expanded_unl.reshape(B * M * K, -1)
        
        unl_goals_expanded = jnp.expand_dims(unl_goals, 1)  # [B, 1, K, goal_dim]
        unl_goals_expanded = jnp.tile(unl_goals_expanded, (1, M, 1, 1))  # [B, M, K, goal_dim]
        unl_goals_expanded = unl_goals_expanded.reshape(B * M * K, -1)
        
        #target_unl_all = state.apply_fn(state.target_params, skip_states_expanded_unl, unl_goals_expanded)
        target_unl_all = self.network.select('target_value')(skip_states_expanded_unl, unl_goals_expanded)
        target_unl_all = target_unl_all.reshape(B, M, K)
        target_unl_max = jnp.max(target_unl_all, axis=1)  # [B, K]
        
        cons_unl = jnp.mean((target_unl_max - pred_unl) ** 2)
        consistency_loss = 0.5 * (cons_pos + cons_unl)

        # === TOTAL LOSS (matching the commented-out version) ===
        total_loss = rank_loss + lambda_cons * consistency_loss
        # + lambda_goal * goal_loss + lambda_mass * mass_loss  (commented out)
        
        metrics = {
            "loss_total": total_loss,
            "loss_rank": rank_loss,
            "loss_cons": consistency_loss,
            "pred_pos": jnp.mean(pred_pos),
            "pred_unl": jnp.mean(pred_unl),
        }
        
        return total_loss, metrics
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        rws_loss, rws_info = self.reach_loss(batch, grad_params)
        for k, v in rws_info.items():
            info[f'rws/{k}'] = v

        loss = rws_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
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

    def predict_reachability(self, states, goals, use_target: bool = False):
        """Return reachability scores in numpy format for visualization/evaluation."""
        module_name = 'target_value' if use_target else 'value'
        value_fn = self.network.select(module_name)
        preds = value_fn(jnp.asarray(states), jnp.asarray(goals))
        return np.asarray(preds)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
          
        # Define value networks.
        value_def = RWSValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

      
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

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='rws',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='ReachabilityGCDataset',  # Dataset class name.
            value_p_curgoal=0.,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
