from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, TDInfoNCECritic


class CRLInfoNCEAgent(flax.struct.PyTreeNode):
    """Contrastive RL agent with TD-InfoNCE style critic.
    
    This agent follows CRL's contrastive learning approach but adapts it for
    the TD-InfoNCE critic architecture that takes (s, a, g, s').
    """
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute contrastive loss following CRL's approach with TD-InfoNCE critic."""
        batch_size = batch['observations'].shape[0]
        
        # TD-InfoNCE critic computes Q(s,a,g,s') = phi(s,a,g)^T @ psi(s')
        # It returns a matrix Q[i,j] = Q(s_i, a_i, g_i, s'_j)
        if module_name == 'critic':
            # For critic, we use actions
            logits = self.network.select(module_name)(
                batch['observations'],
                batch['future_goals'],#batch['value_goals'],
                batch['actions'],
                batch['intermediate_future_goals'],  # These are the future states
                params=grad_params,
            )


            self_logits = self.network.select(module_name)(
                batch['observations'],
                batch['future_goals'],#batch['value_goals'],
                batch['actions'],
                batch['future_goals'],  # These are the future states
                params=grad_params,
            )
        else:
            # For value network (if using AWR), we don't use actions
            # Create dummy actions of zeros
            dummy_actions = jnp.zeros_like(batch['actions'])
            logits = self.network.select(module_name)(
                batch['observations'],
                batch['future_goals'],#batch['value_goals'],
                dummy_actions,
                batch['intermediate_future_goals'],
                params=grad_params,
            )
            sef_logits = self.network.select(module_name)(
                batch['observations'],
                batch['future_goals'],#batch['value_goals'],
                dummy_actions,
                batch['future_goals'],
                params=grad_params,
            )
        
        # logits shape is (B, B, E) if ensemble, (B, B) otherwise
        # Normalize by sqrt of dimension like in CRL
        if len(logits.shape) == 3:  # Ensemble
            # Average over ensemble dimension for loss computation
            logits_for_loss = logits
            self_logits_for_loss = self_logits

        else:
            # Add ensemble dimension for consistency
            logits_for_loss = logits[..., None]
            self_logits_for_loss = self_logits[..., None]

        
        # Apply scaling like CRL (already done inside TDInfoNCECritic if repr_norm=True)
        # The diagonal elements are positive pairs, off-diagonal are negative pairs
        I = jnp.eye(batch_size)
        
        # Compute contrastive loss exactly like CRL
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits_for_loss)
        contrastive_loss = jnp.mean(contrastive_loss)

        self_contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(self_logits_for_loss)
        self_contrastive_loss = jnp.mean(self_contrastive_loss)
        
        # Compute statistics
        logits_mean = jnp.mean(logits_for_loss, axis=-1)  # Average over ensemble
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits_mean * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits_mean * (1 - I)) / jnp.sum(1 - I)
        
        # Extract diagonal values as "value" for statistics (like CRL's v)
        v = jnp.diag(logits_mean)
        v_exp = jnp.exp(v)  # Exponentiate for consistency with CRL
        
        return contrastive_loss+self_contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v_exp.mean(),
            'v_max': v_exp.max(),
            'v_min': v_exp.min(),
            'binary_accuracy': jnp.mean((logits_mean > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits_mean.mean(),
        }
    
    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss following CRL exactly."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss
            # Get V values (using dummy actions for value network)
            dummy_actions = jnp.zeros_like(batch['actions'])
            v_logits = self.network.select('value')(
                batch['observations'],
                batch['actor_goals'],
                dummy_actions,
                batch['actor_goals'],  # Use goal as target state
            )
            # Extract diagonal as V(s,g)
            if len(v_logits.shape) == 2:
                v = jnp.diag(v_logits)
            else:  # Should not have ensemble for value
                v = jnp.diag(v_logits)
            
            # Get Q values
            q_logits = self.network.select('critic')(
                batch['observations'],
                batch['actor_goals'],
                batch['actions'],
                batch['actor_goals'],  # Use goal as target state
            )
            
            # Extract diagonal as Q(s,a,g)
            if len(q_logits.shape) == 3:  # Ensemble
                q1 = jnp.diag(q_logits[..., 0])
                q2 = jnp.diag(q_logits[..., 1])
                q = jnp.minimum(q1, q2)
            else:
                q = jnp.diag(q_logits)
            
            adv = q - v
            
            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)
            
            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                params=grad_params
            )
            log_prob = dist.log_prob(batch['actions'])
            
            actor_loss = -(exp_a * log_prob).mean()
            
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update({
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                })
            
            return actor_loss, actor_info
            
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss
            assert not self.config['discrete']
            
            dist = self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                params=grad_params
            )
            
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            
            # Get Q values for the actor's actions
            q_logits = self.network.select('critic')(
                batch['observations'],
                batch['actor_goals'],
                q_actions,
                batch['actor_goals'],  # Use goal as target state
            )
            
            # Extract diagonal Q values
            if len(q_logits.shape) == 3:  # Ensemble
                q1 = jnp.diag(q_logits[..., 0])
                q2 = jnp.diag(q_logits[..., 1])
                q = jnp.minimum(q1, q2)
            else:
                q = jnp.diag(q_logits)
            
            # Normalize Q values exactly like CRL
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            
            actor_loss = q_loss + bc_loss
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss following CRL exactly."""
        info = {}
        rng = rng if rng is not None else self.rng
        
        # Critic contrastive loss
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        # Value contrastive loss (only for AWR)
        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0
        
        # Actor loss
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        loss = critic_loss + value_loss + actor_loss
        return loss, info
    
    @jax.jit
    def update(self, batch):
        """Update the agent following CRL."""
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
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
    
    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new CRL-InfoNCE agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        ex_goals = ex_observations
        ex_future_states = ex_observations  # Example future states
        
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value'] = GCEncoder(concat_encoder=encoder_module())
        
        # Define critic using TDInfoNCECritic
        critic_def = TDInfoNCECritic(
            hidden_dims=config['value_hidden_dims'],
            repr_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,  # Twin Q for critic
            repr_norm=config.get('repr_norm', True),
            repr_norm_temp=config.get('repr_norm_temp', 1.0),
            gc_encoder=encoders.get('critic'),
        )
        
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions, ex_future_states)),
        )
        
        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network
            # We'll use TDInfoNCECritic without ensemble for value
            value_def = TDInfoNCECritic(
                hidden_dims=config['value_hidden_dims'],
                repr_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,  # No ensemble for V
                repr_norm=config.get('repr_norm', True),
                repr_norm_temp=config.get('repr_norm_temp', 1.0),
                gc_encoder=encoders.get('value'),
            )
            # Value network uses dummy actions
            dummy_actions = jnp.zeros_like(ex_actions)
            network_info['value'] = (value_def, (ex_observations, ex_goals, dummy_actions, ex_future_states))
        
        # Actor definition
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
        
        network_info['actor'] = (actor_def, (ex_observations, ex_goals))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Configuration following CRL with TD-InfoNCE architecture."""
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters following CRL
            agent_name='crl_infonce',
            lr=3e-4,
            batch_size=1024,  # CRL uses 1024
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            latent_dim=512,  # Representation dimension
            layer_norm=True,
            discount=0.99,
            actor_loss='ddpgbc',  # or 'awr'
            alpha=0.1,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            
            # TD-InfoNCE specific
            repr_norm=True,  # Normalize representations
            repr_norm_temp=1.0,  # Temperature for normalization
            
            # Dataset hyperparameters following CRL
            dataset_class='TDInfoNCEDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config