from typing import Sequence, Callable, Tuple, Optional
import numpy as np
import torch.nn as nn
import torch
import torch.distributions as D


class ActorCritic(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        target_critics: nn.ModuleList,
    ):
        super().__init__()
        self.actor = actor
        self.critics = critics
        self.target_critics = target_critics

    def forward(self, x):
        raise NotImplementedError

    def get_action_distribution(self, observations: torch.Tensor) -> D.Distribution:
        """
        Compute an action distribution for a given observation.
        """
        means, log_stds = self.actor(observations)
        action_dist = D.Normal(means, torch.exp(log_stds))
        action_dist = D.TransformedDistribution(
            action_dist, D.TanhTransform(cache_size=1)
        )
        action_dist = D.Independent(action_dist, reinterpreted_batch_ndims=1)
        return action_dist

    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute an action for a given observation.
        Output shape: (batch_size, action_dim)
        """
        action_dist = self.get_action_distribution(observations)
        actions = action_dist.sample()
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

    def __repr__(self):
        return f"{self.__class__.__name__}"
