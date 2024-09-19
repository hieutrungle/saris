from typing import Tuple
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

    # def get_action_distribution(self, observations: torch.Tensor) -> D.Distribution:
    #     """
    #     Compute an action distribution for a given observation.
    #     """
    #     raise NotImplementedError

    @staticmethod
    def get_actions(
        self, observations: torch.Tensor, train: bool = False
    ) -> torch.Tensor:
        """
        Compute an action for a given observation.

        Return:
            actions: (batch_size, action_dim)
        """
        raise NotImplementedError

    @staticmethod
    def get_target_q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target Q-values for a given observation-action pair.

        Output shape: (num_critics, batch_size)
        """
        raise NotImplementedError

    @staticmethod
    def get_q_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for a given observation-action pair.
        Output shape: (num_critics, batch_size)
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"


class SoftActorCritic(ActorCritic):

    def __init__(
        self,
        actor: nn.Module,
        critics: nn.ModuleList,
        target_critics: nn.ModuleList,
    ):
        super().__init__(actor, critics, target_critics)

    def get_actions(
        self, observations: torch.Tensor, train: bool = False
    ) -> torch.Tensor:
        """
        Compute an action for a given observation.

        Return:
            actions: (batch_size, action_dim)

        actor sample returns:
            actions: (num_samples, batch_size, action_dim)
            log_probs: (batch_size,)
            means: (batch_size, action_dim)
        """
        if train:
            actions, _, _ = self.actor.sample(observations)
            actions = actions[0]
        else:
            _, _, actions = self.actor.sample(observations)
        return actions

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
        return target_q_values

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
        return q_values

    def __repr__(self):
        return f"{self.__class__.__name__}"
