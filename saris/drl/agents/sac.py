from saris.drl.agents.actor_critic import ActorCritic
import torch.nn as nn


# @jax.tree_util.register_pytree_node_class
# @dataclass
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
