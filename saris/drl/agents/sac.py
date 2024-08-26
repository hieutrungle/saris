from typing import Sequence, Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from saris.drl.infrastructure.train_state import TrainState
from saris import distributions as D
from flax import struct
import numpy as np
from dataclasses import dataclass
from saris.drl.agents.actor_critic import ActorCritic


@jax.tree_util.register_pytree_node_class
@dataclass
class SoftActorCritic(ActorCritic):

    actor_state: TrainState
    critic_states: Sequence[TrainState]
    target_critic_states: Sequence[TrainState]
    alpha_state: TrainState

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def tree_flatten(self):
        # first group (if it's non-hashable/dynamic)
        # or the second group (if it's hashable/static)

        # arrays / dynamic values
        # children = (self.actor_state, self.critic_states, self.target_critic_states)
        children = tuple(
            [
                self.alpha_state,
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
