from typing import Sequence, Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from saris.drl.infrastructure.train_state import TrainState
from saris import distributions as D
from flax import struct
import numpy as np
import dataclasses
from dataclasses import dataclass


@jax.tree_util.register_pytree_node_class
@dataclass
class ActorCritic:

    actor_state: TrainState
    critic_states: Sequence[TrainState]
    target_critic_states: Sequence[TrainState]

    def get_action_distribution(
        self,
        observations: np.ndarray,
        actor_params: struct.PyTreeNode,
        actor_apply_fn: Callable,
    ) -> D.Distribution:
        """
        Compute an action distribution for a given observation.
        """
        means, log_stds = actor_apply_fn({"params": actor_params}, observations)
        action_dist = D.Normal(means, jnp.exp(log_stds))
        action_dist = D.Transformed(action_dist, D.Tanh())
        action_dist = D.Independent(action_dist, reinterpreted_batch_ndims=1)
        return action_dist

    @jax.jit
    def _get_actions(
        self,
        observations: np.ndarray,
        actor_state: TrainState,
        key: jax.random.PRNGKey,
    ) -> np.ndarray:
        action_dist = self.get_action_distribution(
            observations, actor_state.params, actor_state.apply_fn
        )
        actions = action_dist.sample(seed=key)
        return actions

    def get_actions(self, observations: np.ndarray) -> jnp.ndarray:
        """
        Compute an action for a given observation.
        Output shape: (batch_size, action_dim)
        """
        actions = self._get_actions(
            observations, self.actor_state, self.actor_state.rng
        )
        return actions

    def get_q_values(
        self,
        tuple_critic_params: Tuple[struct.PyTreeNode],
        critic_apply_fns: Tuple[Callable],
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Q-values for a given observation-action pair.
        Output shape: (num_critics, batch_size)
        """

        def _get_q_values_loop():
            q_values = []
            for i, critic_params in enumerate(tuple_critic_params):
                q_values.append(
                    critic_apply_fns[i](
                        {"params": critic_params}, observations, actions
                    )
                )
            return jnp.stack(q_values, axis=0)

        q_values = _get_q_values_loop()

        return q_values.squeeze()

    def get_entropy(
        self,
        action_distribution: D.Distribution,
        sample_shape: Sequence[int] = (32,),
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> jnp.ndarray:
        _, log_probs = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=sample_shape
        )

        entropy_est = -jnp.mean(log_probs, axis=0)
        return entropy_est

    def replace(self, **updates):
        """Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def tree_flatten(self):
        # first group (if it's non-hashable/dynamic)
        # or the second group (if it's hashable/static)

        # arrays / dynamic values
        # children = (self.actor_state, self.critic_states, self.target_critic_states)
        children = tuple(
            [
                self.actor_state,
                self.critic_states,
                self.target_critic_states,
            ]
        )

        # static values
        aux_data = tuple([])
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
