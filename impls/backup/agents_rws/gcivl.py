import copy
import json
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.rws import RWSAgent, get_config as get_rws_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCActor, GCDiscreteActor, GCValue


class GCIVLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.

    This is a variant of GCIQL that only uses a V function, without Q functions.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    rws_agent: Any = nonpytree_field(default=None)
    rws_expected_action_dim: Any = nonpytree_field(default=None)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _compute_batch_weights(self, batch):
        """Compute reachability weights for the batch."""
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']

        batch_size = observations.shape[0]
        base_weights = jnp.full((batch_size,), 1.0 / batch_size)

        if not self.config.get('load_rws', False) or self.rws_agent is None:
            return base_weights

        observations = jnp.asarray(observations, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        goals = jnp.asarray(goals, dtype=jnp.float32)

        expected_action_dim = self.rws_expected_action_dim
        if expected_action_dim is not None:
            actual_action_dim = actions.shape[-1]
            if actual_action_dim != expected_action_dim:
                if expected_action_dim == 1:
                    actions = jnp.mean(actions, axis=-1, keepdims=True)
                elif actual_action_dim > expected_action_dim:
                    actions = actions[..., :expected_action_dim]
                else:
                    pad_shape = actions.shape[:-1] + (expected_action_dim - actual_action_dim,)
                    padding = jnp.zeros(pad_shape, dtype=actions.dtype)
                    actions = jnp.concatenate([actions, padding], axis=-1)

        scores = self.rws_agent.network.select('critic')(observations, goals, actions)
        scores = jnp.squeeze(scores, axis=-1)

        weighting_mode = self.config.get('reachability_weighting', 'normalized').lower()

        if weighting_mode == 'vanilla':
            raw = jnp.clip(scores, a_min=0.0)
            raw_sum = jnp.sum(raw)
            denom = jnp.maximum(raw_sum, 1e-9)
            weights = jnp.where(raw_sum > 1e-9, raw / denom, base_weights)
        else:
            epsilon = self.config.get('reachability_floor', 0.05)

            if weighting_mode == 'adv':
                value_scores = self.rws_agent.network.select('value')(observations, goals)
                value_scores = jnp.squeeze(value_scores, axis=-1)
                raw = jnp.exp(scores - value_scores)
            else:
                temperature = self.config.get('reachability_temperature', 1.0)
                raw = jnp.exp(scores / temperature)

            raw_sum = jnp.maximum(jnp.sum(raw), 1e-9)
            normalized = raw / raw_sum
            weights = (1.0 - epsilon) * normalized + epsilon / batch_size

        return weights

    def value_loss(self, batch, grad_params, weights):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        per_sample1 = self.expectile_loss(adv, q1 - v1, self.config['expectile'])
        per_sample2 = self.expectile_loss(adv, q2 - v2, self.config['expectile'])
        value_loss1 = jnp.sum(weights * per_sample1)
        value_loss2 = jnp.sum(weights * per_sample2)
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, weights, rng=None):
        """Compute the AWR actor loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        combined_weights = weights * exp_a
        actor_loss = -jnp.sum(combined_weights * log_prob)

        actor_info = {
            'actor_loss': actor_loss,
            'adv': jnp.sum(weights * adv),
            'bc_log_prob': jnp.sum(weights * log_prob),
        }
        if not self.config['discrete']:
            mse_per_sample = jnp.mean((dist.mode() - batch['actions']) ** 2, axis=-1)
            scale_diag = dist.scale_diag
            if scale_diag.ndim > 1:
                std_metric = jnp.sum(weights * jnp.mean(scale_diag, axis=-1))
            else:
                std_metric = jnp.mean(scale_diag)
            actor_info.update(
                {
                    'mse': jnp.sum(weights * mse_per_sample),
                    'std': std_metric,
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        weights = self._compute_batch_weights(batch)

        value_loss, value_info = self.value_loss(batch, grad_params, weights)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, weights, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        info['reachability/weight_mean'] = weights.mean()
        info['reachability/weight_max'] = weights.max()
        info['reachability/weight_min'] = weights.min()

        loss = value_loss + actor_loss
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

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @staticmethod
    def _initialize_rws_agent(seed, ex_observations, ex_actions, config):
        """Load a pretrained RWS agent if requested."""
        load_path = config.get('load_rws_path')
        load_epoch = config.get('load_rws_epoch')

        if load_path is None or load_epoch is None:
            raise ValueError('Both `load_rws_path` and `load_rws_epoch` must be specified when `load_rws` is enabled.')

        load_path = os.path.expanduser(load_path)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'RWS checkpoint directory not found: {load_path}')
        load_epoch = int(load_epoch)
        rws_config = get_rws_config()

        flags_path = os.path.join(load_path, 'flags.json')
        if os.path.exists(flags_path):
            with open(flags_path, 'r') as f:
                saved_flags = json.load(f)
            saved_agent_cfg = saved_flags.get('agent', {})
            if isinstance(saved_agent_cfg, dict):
                for key, value in saved_agent_cfg.items():
                    rws_config[key] = value

        if rws_config.get('frame_stack') is None and config.get('frame_stack') is not None:
            rws_config['frame_stack'] = config['frame_stack']
        if rws_config.get('encoder') is None and config.get('encoder') is not None:
            rws_config['encoder'] = config['encoder']

        rws_agent = RWSAgent.create(seed, ex_observations, ex_actions, rws_config)
        rws_agent = restore_agent(rws_agent, load_path, load_epoch)

        expected_action_dim = None
        params = rws_agent.network.params
        critic_params = params.get('modules_critic')
        value_params = params.get('modules_value')
        if critic_params is not None and value_params is not None:
            fc0 = critic_params.get('fc0')
            value_fc0 = value_params.get('fc0')
            if fc0 is not None and value_fc0 is not None:
                critic_input_dim = fc0['kernel'].shape[0]
                value_input_dim = value_fc0['kernel'].shape[0]
                inferred = critic_input_dim - value_input_dim
                if inferred >= 0:
                    expected_action_dim = inferred

        return rws_agent, expected_action_dim

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

        load_rws_flag = bool(config.get('load_rws', False))
        load_rws_path = config.get('load_rws_path', None)
        if not load_rws_flag and load_rws_path is not None:
            load_rws_flag = True
        config['load_rws'] = load_rws_flag

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
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('value'),
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        rws_agent = None
        rws_expected_action_dim = None
        if load_rws_flag:
            rws_agent, rws_expected_action_dim = cls._initialize_rws_agent(seed, ex_observations, ex_actions, config)
            config['rws_expected_action_dim'] = rws_expected_action_dim

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            rws_agent=rws_agent,
            rws_expected_action_dim=rws_expected_action_dim,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcivl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            load_rws=False,  # Whether to load a pretrained reachability model for weighting.
            load_rws_path=ml_collections.config_dict.placeholder(str),  # Path to the pretrained RWS checkpoint.
            load_rws_epoch=ml_collections.config_dict.placeholder(int),  # Epoch of the pretrained RWS checkpoint.
            reachability_weighting='normalized',  # RWS weighting mode: 'vanilla', 'normalized', 'adv'.
            reachability_temperature=1.0,  # Temperature for reachability weighting.
            reachability_floor=0.05,  # Coverage floor for reachability weighting.
            rws_expected_action_dim=None,
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.5,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
