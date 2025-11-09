import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, TDInfoNCECritic


class TDInfoNCEAgent(flax.struct.PyTreeNode):
    """Temporal Difference InfoNCE (TD-InfoNCE) agent.
    
    This agent implements the TD-InfoNCE algorithm from:
    "Contrastive Difference Predictive Coding" (Zheng et al., 2024)
    
    Key features:
    - Contrastive critic: Q(s,a,g,s_future) with twin Q networks
    - TD learning with importance sampling for off-policy reasoning
    - DDPG+BC style actor loss (following CRL)
    """
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD-InfoNCE critic loss.
        
        Following Eq. 10 in the TD-InfoNCE paper:
        L = (1-γ) * L_next + γ * L_future
        
        L_next: Cross-entropy loss for predicting next state
        L_future: Cross-entropy loss with importance-weighted labels
        
        Args:
            batch: Batch with keys:
                - observations (N, obs_dim)
                - actions (N, act_dim)
                - value_goals (N, goal_dim): random goals
                - next_observations (N, obs_dim): positive future states
                - random_goals (N, goal_dim): negative future states (rolled)
            grad_params: Parameters for gradient computation.
            rng: Random key.
        
        Returns:
            Tuple of (loss, info_dict).
        """
        batch_size = batch['observations'].shape[0]
        
        # ===== Term 1: Next state prediction (positive term) =====
        # Q(s, a, g, s') where s' is the actual next state
        # Shape: (N, N, 2)
        q_next = self.network.select('critic')(
            batch['observations'],
            batch['future_goals'], #batch['value_goals'],  # random goals
            batch['actions'],
            batch['next_observations'],  # next states as future states
            params=grad_params
        )
        
        # Diagonal elements are positive pairs: Q[i, i] = Q(s_i, a_i, g_i, s'_i)
        # Off-diagonal are negatives
        I = jnp.eye(batch_size)
        I_expanded = I[:, :, None].repeat(q_next.shape[-1], axis=-1)  # (N, N, 2)
        
        # Cross-entropy loss across each twin Q
        loss_next = jax.vmap(
            lambda logits: optax.softmax_cross_entropy(logits=logits, labels=I),
            in_axes=-1, out_axes=-1
        )(q_next)  # (N, 2)
        loss_next = jnp.mean(loss_next)
        
        # ===== Term 2: Future state prediction with importance sampling =====
        # Sample next actions from policy
        rng, actor_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(
            batch['next_observations'],
            batch['future_goals'],
            temperature=1.0
        )
        next_actions = next_dist.sample(seed=actor_rng)
        if not self.config['discrete']:
            next_actions = jnp.clip(next_actions, -1, 1)
        
        # Compute importance weights using target critic
        # w[i,j] ∝ exp(Q_target(s'_i, a'_i, g_i, rand_g_j))
        # Shape: (N, N, 2)
        logits_w = self.network.select('target_critic')(
            batch['next_observations'],
            batch['future_goals'],
            next_actions,
            batch['random_goals']  # rolled random goals as negative future states
        )
        
        # Take minimum across twin Q for importance weights
        logits_w_min = jnp.min(logits_w, axis=-1)  # (N, N)
        
        # Compute softmax weights (row-wise)
        w = jax.nn.softmax(logits_w_min, axis=1)  # (N, N)
        w = jax.lax.stop_gradient(w)  # Stop gradient for weights
        w_expanded = w[:, :, None].repeat(q_next.shape[-1], axis=-1)  # (N, N, 2)
        
        # Compute Q(s, a, g, rand_g) for negative future states
        # Shape: (N, N, 2)
        q_future = self.network.select('critic')(
            batch['observations'],
            batch['future_goals'],
            batch['actions'],
            batch['random_goals'],  # rolled random goals
            params=grad_params
        )
        
        # Cross-entropy with importance-weighted labels
        loss_future = jax.vmap(
            lambda logits, labels: optax.softmax_cross_entropy(logits=logits, labels=labels),
            in_axes=-1, out_axes=-1
        )(q_future, w_expanded)  # (N, 2)
        loss_future = jnp.mean(loss_future)
        
        # ===== Total critic loss =====
        critic_loss = (1 - self.config['discount']) * loss_next + \
                      self.config['discount'] * loss_future
        
        # ===== Detailed Logging (matching CRL and original TD-InfoNCE) =====
        # Logits statistics
        q_next_pos = jnp.mean(jax.vmap(jnp.diag, -1, -1)(q_next))  # Mean diagonal across both Qs
        q_next_pos1 = jnp.mean(jnp.diag(q_next[..., 0]))  # First Q diagonal
        q_next_pos2 = jnp.mean(jnp.diag(q_next[..., 1]))  # Second Q diagonal
        q_next_neg = jnp.mean(q_next * (1 - I_expanded))  # Off-diagonal (negatives)
        
        q_future_pos = jnp.mean(jax.vmap(jnp.diag, -1, -1)(q_future))
        q_future_pos1 = jnp.mean(jnp.diag(q_future[..., 0]))
        q_future_pos2 = jnp.mean(jnp.diag(q_future[..., 1]))
        q_future_neg = jnp.mean(q_future * (1 - I_expanded))
        
        # Entropy of logits distributions
        q_next_probs = jax.nn.softmax(q_next[..., 0], axis=1)
        q_next_entropy = -jnp.sum(q_next_probs * jax.nn.log_softmax(q_next[..., 0], axis=1), axis=1)
        
        q_future_probs = jax.nn.softmax(q_future[..., 0], axis=1)
        q_future_entropy = -jnp.sum(q_future_probs * jax.nn.log_softmax(q_future[..., 0], axis=1), axis=1)
        
        logits_w_probs = jax.nn.softmax(logits_w_min, axis=1)
        logits_w_entropy = -jnp.sum(logits_w_probs * jax.nn.log_softmax(logits_w_min, axis=1), axis=1)
        
        # Accuracy metrics (like CRL)
        q_next_avg = jnp.mean(q_next, axis=-1)  # Average over twin Qs
        binary_accuracy_next = jnp.mean((q_next_avg > 0) == I)
        categorical_accuracy_next = jnp.mean(jnp.argmax(q_next_avg, axis=1) == jnp.argmax(I, axis=1))
        
        q_future_avg = jnp.mean(q_future, axis=-1)
        binary_accuracy_future = jnp.mean((q_future_avg > 0) == I)
        categorical_accuracy_future = jnp.mean(jnp.argmax(q_future_avg, axis=1) == jnp.argmax(I, axis=1))
        
        return critic_loss, {
            # Main losses
            'critic_loss': critic_loss,
            'loss_next': loss_next,
            'loss_future': loss_future,
            
            # Q_next statistics (positive term)
            'q_next_pos': q_next_pos,  # Mean diagonal
            'q_next_pos1': q_next_pos1,  # First Q diagonal
            'q_next_pos2': q_next_pos2,  # Second Q diagonal
            'q_next_neg': q_next_neg,  # Off-diagonal
            'q_next_mean': jnp.mean(q_next),  # Overall mean
            'q_next_max': jnp.max(q_next),
            'q_next_min': jnp.min(q_next),
            'q_next_entropy': jnp.mean(q_next_entropy),
            
            # Q_future statistics (negative term)
            'q_future_pos': q_future_pos,
            'q_future_pos1': q_future_pos1,
            'q_future_pos2': q_future_pos2,
            'q_future_neg': q_future_neg,
            'q_future_mean': jnp.mean(q_future),
            'q_future_max': jnp.max(q_future),
            'q_future_min': jnp.min(q_future),
            'q_future_entropy': jnp.mean(q_future_entropy),
            
            # Importance weights statistics
            'w_diag': jnp.mean(jnp.diag(w)),  # Diagonal weights
            'w_mean': jnp.mean(w),  # Overall mean
            'w_max': jnp.max(w),
            'w_min': jnp.min(w),
            'logits_w_entropy': jnp.mean(logits_w_entropy),
            
            # Accuracy metrics (like CRL)
            'binary_accuracy_next': binary_accuracy_next,
            'categorical_accuracy_next': categorical_accuracy_next,
            'binary_accuracy_future': binary_accuracy_future,
            'categorical_accuracy_future': categorical_accuracy_future,
        }
    
    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss using DDPG+BC style (following CRL).
        
        Following TD-InfoNCE's actor objective (Eq. 11) combined with CRL's DDPG+BC:
        - Q loss: maximize Q(s, π(s,g), g, g) to reach goal g
        - BC loss: maximize log π(a|s,g) for behavioral cloning regularization
        
        Args:
            batch: Batch with keys:
                - observations (N, obs_dim)
                - actions (N, act_dim): behavioral actions
                - actor_goals (N, goal_dim): future or random goals
            grad_params: Parameters for gradient computation.
            rng: Random key.
        
        Returns:
            Tuple of (loss, info_dict).
        """
        batch_size = batch['observations'].shape[0]
        
        # ===== Q loss: maximize Q(s, π(s,g), g, g) =====
        # Sample actions from current policy
        dist = self.network.select('actor')(
            batch['observations'],
            batch['actor_goals'],
            params=grad_params,
            temperature=1.0
        )
        
        if self.config['const_std']:
            q_actions = dist.mode()  # Use mean for deterministic policy
        else:
            q_actions = dist.sample(seed=rng)
        
        if not self.config['discrete']:
            q_actions = jnp.clip(q_actions, -1, 1)
        
        # Compute Q(s, π(s,g), g, g)
        # The agent wants to reach goal g, so future_state = g
        # Shape: (N, N, 2)
        q = self.network.select('critic')(
            batch['observations'],
            batch['actor_goals'],
            q_actions,
            batch['actor_goals']  # future_state = goal (reaching the goal!)
        )
        
        # Take minimum across twin Q
        q_min = jnp.min(q, axis=-1)  # (N, N)
        
        # Diagonal elements Q[i,i] = Q(s_i, π(s_i,g_i), g_i, g_i)
        I = jnp.eye(batch_size)
        
        # Maximize Q by minimizing cross-entropy
        q_loss_raw = optax.softmax_cross_entropy(logits=q_min, labels=I)
        q_loss = -q_loss_raw.mean() / jax.lax.stop_gradient(jnp.abs(q_loss_raw).mean() + 1e-6)
        
        # ===== BC loss: regularize with behavioral cloning =====
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -(self.config['alpha'] * log_prob).mean()
        
        # ===== Total actor loss =====
        actor_loss = q_loss + bc_loss
        
        # ===== Detailed Logging (matching CRL) =====
        # Q statistics
        q_diag = jnp.diag(q_min)  # Diagonal Q values
        q_off_diag = q_min * (1 - I)  # Off-diagonal Q values
        
        # Accuracy metrics
        binary_accuracy = jnp.mean((q_min > 0) == I)
        categorical_accuracy = jnp.mean(jnp.argmax(q_min, axis=1) == jnp.argmax(I, axis=1))
        
        # Entropy of Q distribution
        q_probs = jax.nn.softmax(q_min, axis=1)
        q_entropy = -jnp.sum(q_probs * jax.nn.log_softmax(q_min, axis=1), axis=1)
        
        actor_info = {
            # Main losses
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            
            # Q statistics (matching CRL)
            'q_mean': jnp.mean(q_min),
            'q_abs_mean': jnp.abs(q_min).mean(),
            'q_diag_mean': jnp.mean(q_diag),  # Diagonal (positive pairs)
            'q_off_diag_mean': jnp.mean(q_off_diag),  # Off-diagonal (negatives)
            'q_max': jnp.max(q_min),
            'q_min': jnp.min(q_min),
            'q_std': jnp.std(q_min),
            
            # Individual twin Q statistics
            'q1_mean': jnp.mean(q[..., 0]),
            'q2_mean': jnp.mean(q[..., 1]),
            'q1_diag_mean': jnp.mean(jnp.diag(q[..., 0])),
            'q2_diag_mean': jnp.mean(jnp.diag(q[..., 1])),
            
            # BC statistics (matching CRL)
            'bc_log_prob': log_prob.mean(),
            'bc_log_prob_std': jnp.std(log_prob),
            
            # Accuracy metrics
            'binary_accuracy': binary_accuracy,
            'categorical_accuracy': categorical_accuracy,
            
            # Q distribution entropy
            'q_entropy': jnp.mean(q_entropy),
        }
        
        if not self.config['discrete']:
            # Additional continuous action statistics
            actor_info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'mse_std': jnp.std((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
                'action_mean': jnp.mean(jnp.abs(q_actions)),
                'action_max': jnp.max(jnp.abs(q_actions)),
            })
        else:
            # Discrete action statistics
            actor_info.update({
                'action_entropy': -jnp.mean(jnp.sum(
                    jnp.exp(dist.logits) * dist.logits, axis=-1
                )),
            })
        
        return actor_loss, actor_info
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)
        
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        loss = critic_loss + actor_loss
        return loss, info
    
    def target_update(self, network, module_name):
        """Update the target network using exponential moving average."""
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
        self.target_update(new_network, 'critic')
        
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
        """Create a new TD-InfoNCE agent.
        
        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define encoders for visual observations
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_obs'] = encoder_module()
            encoders['critic_future'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
        
        # Define TD-InfoNCE critic
        critic_def = TDInfoNCECritic(
            hidden_dims=config['value_hidden_dims'],
            repr_dim=config['repr_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,  # Twin Q
            repr_norm=config['repr_norm'],
            repr_norm_temp=config['repr_norm_temp'],
            gc_encoder=encoders.get('critic_obs'),
        )
        
        target_critic_def = TDInfoNCECritic(
            hidden_dims=config['value_hidden_dims'],
            repr_dim=config['repr_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            repr_norm=config['repr_norm'],
            repr_norm_temp=config['repr_norm_temp'],
            gc_encoder=encoders.get('critic_future'),
        )
        
        # Define actor
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
        
        # Initialize networks
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions, ex_goals)),
            target_critic=(target_critic_def, (ex_observations, ex_goals, ex_actions, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        
        # Copy critic params to target critic
        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters (following TD-InfoNCE paper).
            agent_name='td_infonce',  # Agent name.
            lr=3e-4,  # Learning rate (same as TD-InfoNCE).
            batch_size=1024,  # Batch size (TD-InfoNCE uses 256).
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            repr_dim=16,  # Representation dimension (TD-InfoNCE uses 16).
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=0.1,  # BC coefficient in DDPG+BC (following CRL).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            repr_norm=True,  # Whether to normalize representations (TD-InfoNCE uses True).
            repr_norm_temp=1.0,  # Temperature for normalized representations.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters (following TD-InfoNCE sampling strategy).
            dataset_class='TDInfoNCEDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Not used (compatibility).
            value_p_trajgoal=1.,  # Not used (compatibility).
            value_p_randomgoal=0.,  # Not used (compatibility).
            value_geom_sample=True,  # Use geometric sampling for future goals.
            actor_p_curgoal=0.0,  # Not used.
            actor_p_trajgoal=1.,  # Use future goals from trajectory for actor.
            actor_p_randomgoal=0.,  # Don't use random goals for actor (TD-InfoNCE default).
            actor_geom_sample=True,  # Use geometric sampling.
            gc_negative=False,  # Not used in TD-InfoNCE critic.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config