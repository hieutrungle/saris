from saris.drl.agents.actor_critic import ActorCritic
import torch.nn as nn
import torch
import torch.distributions as D
from typing import Tuple


class SoftActorCritic(ActorCritic):

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        target_critics: nn.ModuleList,
        alpha: nn.Module,
    ):
        super().__init__(actor, critics, target_critics)
        self.alpha = alpha

    # def get_action_distribution(self, observations: torch.Tensor) -> D.Distribution:
    #     """
    #     Compute an action distribution for a given observation.
    #     """
    #     means, log_stds = self.actor(observations)
    #     action_dist = D.Normal(means, torch.exp(log_stds))
    #     action_dist = D.TransformedDistribution(
    #         action_dist, D.TanhTransform(cache_size=1)
    #     )
    #     action_dist = D.Independent(action_dist, reinterpreted_batch_ndims=1)
    #     return action_dist

    def get_actions(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute an action for a given observation.
        Return:
            actions: (batch_size, action_dim)
            log_probs: (batch_size,)
            means: (batch_size, action_dim)
        """
        action, log_prob, mean = self.actor.sample(observations)
        return action, log_prob, mean

    def get_target_q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target Q-values for a given observation-action pair.
        Output shape: (num_critics, batch_size)
        """
        target_q_values = []
        for i, target_critic in enumerate(self.target_critics):
            target_q_values.append(target_critic(observations, actions))
        target_q_values = torch.stack(target_q_values)
        return target_q_values.squeeze()

    def get_q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for a given observation-action pair.
        Output shape: (num_critics, batch_size)
        """

        q_values = []
        for i, critic in enumerate(self.critics):
            q_values.append(critic(observations, actions))
        q_values = torch.stack(q_values)
        return q_values.squeeze()

    def __repr__(self):
        return f"{self.__class__.__name__}"
