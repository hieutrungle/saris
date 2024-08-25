from typing import Sequence, Callable
import jax
import jax.numpy as jnp
from saris.drl.infrastructure.train_state import TrainState
from saris import distributions as D
from flax import struct
import numpy as np


@jax.tree_util.register_pytree_node_class
class ActorCritic:
    def __init__(
        self,
        actor_state: TrainState,
        critic_states: Sequence[TrainState],
        target_critic_states: Sequence[TrainState],
    ):
        self.actor_state = actor_state
        self.critic_states = critic_states
        self.target_critic_states = target_critic_states

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
    def _get_action(
        self,
        observations: np.ndarray,
        actor_state: TrainState,
        key: jax.random.PRNGKey,
    ) -> np.ndarray:
        action_dist = self.get_action_distribution(
            observations, actor_state.params, actor_state.apply_fn
        )
        action = action_dist.sample(seed=key)
        return jnp.squeeze(action, axis=0)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute an action for a given observation.
        """
        observations = np.expand_dims(observation, axis=0)
        action = self._get_action(observations, self.actor_state, self.actor_state.rng)
        return np.array(action).squeeze()

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
