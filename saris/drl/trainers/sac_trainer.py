from typing import Any, Tuple, Callable, Dict
from saris.drl.trainers import ac_trainer
from saris.drl.agents.actor_critic import ActorCritic
from saris.drl.agents.sac import SoftActorCritic
from saris.drl.networks.alpha import Alpha
import numpy as np
import functools
import copy
import torch
import torch.nn as nn
import torch.distributions as D


class SoftActorCriticTrainer(ac_trainer.ActorCriticTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_entropy = np.asarray(-np.prod(self.action_shape), dtype=np.float32)
        alpha = Alpha(0.05)
        self.summarize_model(alpha, [[1, 1]])

        self.agent = SoftActorCritic(
            actor=self.agent.actor,
            critics=self.agent.critics,
            target_critics=self.agent.target_critics,
            alpha=alpha,
        )

    # def init_agent_optimizer(
    #     self,
    #     agent: SoftActorCritic,
    #     drl_config: Dict[str, Any],
    # ) -> SoftActorCritic:
    #     """
    #     Initializes the optimizer for the agent's components:
    #     - actor
    #     - critics
    #     - target critics
    #     - other components
    #     """
    #     # Initialize optimizer for actor and critic
    #     actor_state = self.init_optimizer(
    #         agent.actor_state,
    #         self.actor_optimizer_hparams,
    #         drl_config["total_steps"],
    #         drl_config["num_train_steps_per_env_step"],
    #     )

    #     critic_states = []
    #     for i in range(self.num_critics):
    #         state = self.init_optimizer(
    #             agent.critic_states[i],
    #             self.critic_optimizer_hparams,
    #             drl_config["total_steps"],
    #             drl_config["num_train_steps_per_env_step"]
    #             * drl_config["num_critic_updates"],
    #         )
    #         critic_states.append(state)

    #     agent = agent.replace(
    #         actor_state=actor_state,
    #         critic_states=critic_states,
    #     )

    #     alpha_optimizer_hparams = copy.deepcopy(self.actor_optimizer_hparams)
    #     alpha_state = self.init_optimizer(
    #         agent.alpha_state,
    #         alpha_optimizer_hparams,
    #         drl_config["total_steps"],
    #         drl_config["num_train_steps_per_env_step"],
    #     )
    #     agent = agent.replace(
    #         alpha_state=alpha_state,
    #     )
    #     return agent

    def create_step_functions(self):

        def accumulate_gradients(agent, batch, rng_key):
            batch_size = batch[0].shape[0]
            num_minibatches = self.grad_accum_steps
            minibatch_size = batch_size // self.grad_accum_steps
            rngs = jax.random.split(rng_key, num_minibatches)
            grad_fn = jax.value_and_grad(actor_loss)

            def _minibatch_step(
                minibatch_idx: jax.Array | int,
            ) -> Tuple[struct.PyTreeNode, jnp.ndarray]:
                """Determine gradients and metrics for a single minibatch."""
                minibatch = jax.tree_map(
                    lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
                        x,
                        start_index=minibatch_idx * minibatch_size,
                        slice_size=minibatch_size,
                        axis=0,
                    ),
                    batch,
                )
                step_loss, step_grads = grad_fn(
                    actor_state.params,
                    minibatch,
                    train=True,
                    rng_key=rngs[minibatch_idx],
                )
                return step_loss, step_grads

            def _scan_step(
                carry: Tuple[struct.PyTreeNode, jnp.ndarray],
                minibatch_idx: jax.Array | int,
            ) -> Tuple[Tuple[struct.PyTreeNode, jnp.ndarray], None]:
                """Scan step function for looping over minibatches."""
                step_loss, step_grads = _minibatch_step(minibatch_idx)
                carry = jax.tree_map(jnp.add, carry, (step_loss, step_grads))
                return carry, None

            # Determine initial shapes for gradients and loss.
            loss_shape, grads_shapes = jax.eval_shape(_minibatch_step, 0)
            grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
            loss = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), loss_shape)

            # Loop over minibatches to determine gradients and metrics.
            (loss, grads), _ = jax.lax.scan(
                _scan_step,
                init=(loss, grads),
                xs=jnp.arange(num_minibatches),
                length=num_minibatches,
            )

            # Average gradients over minibatches.
            grads = jax.tree_map(lambda g: g / num_minibatches, grads)
            return loss, grads

        def do_q_backup(next_qs: torch.Tensor) -> torch.Tensor:
            """
            Handle Q-values from multiple different target critic networks to produce target values.

            Clip-Q, clip to the minimum of the two critics' predictions.

            Parameters:
                next_qs (jnp.ndarray): Q-values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FROM the different critics.
            Returns:
                jnp.ndarray: Target values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FOR the different critics.
            """
            next_qs, _ = torch.min(next_qs, dim=0)
            return next_qs

        def calc_critic_loss(
            batch: dict[str, np.ndarray],
        ) -> dict[str, torch.Tensor]:
            obs, acts, rews, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )

            with torch.no_grad():
                # next_actions shape: (num_actor_samples, batch_size, action_dim)
                next_act_dist: D.Distribution = self.agent.get_action_distribution(
                    next_obs
                )

                next_actions = next_act_dist.sample(
                    sample_shape=(self.num_actor_samples,)
                )
                next_obs = torch.unsqueeze(next_obs, dim=0)
                next_obs = torch.repeat_interleave(
                    next_obs, self.num_actor_samples, dim=0
                )

                # next_q_values shape: (num_critics, num_actor_samples, batch_size)
                next_q_values = self.agent.get_target_q_values(next_obs, next_actions)

                # next_q_values shape: (num_actor_samples, batch_size)
                next_q_values = do_q_backup(next_q_values)
                # next_q_values shape: (batch_size)
                next_q_values = torch.mean(next_q_values, axis=0)

                # Entropy regularization
                # next_action_entropy shape: (batch_size)
                next_action_entropy = self.agent.get_entropy(next_act_dist)
                alpha = self.agent.alpha(
                    torch.tensor(1.0, device=next_action_entropy.device).reshape(1, 1)
                )
                next_q_values = next_q_values + alpha * next_action_entropy

                target_q_values = rews + self.discount * (1.0 - dones) * next_q_values
                target_q_values = torch.unsqueeze(target_q_values, dim=0)
                target_q_values = torch.repeat_interleave(
                    target_q_values, self.num_critics, dim=0
                )

            q_values = self.agent.get_q_values(obs, acts)

            # # Mask out NaN values
            # mask = jnp.isnan(q_values) | jnp.isnan(target_q_values)
            # q_values = jnp.where(mask, 0, q_values)
            # target_q_values = jnp.where(mask, 0, target_q_values)
            crtic_loss = 0.5 * torch.mean((q_values - target_q_values) ** 2)

            critic_info = {
                "q_values": torch.mean(q_values),
                "next_q_values": torch.mean(next_q_values),
                "target_q_values": torch.mean(target_q_values),
                "critic_loss": crtic_loss,
            }
            return crtic_loss, critic_info

        def update_crtics(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:

            loss, info = calc_critic_loss(batch)
            # TODO: optimizer
            return info

        def update_target_crtics() -> None:
            """
            Update target critics with moving average of current critics.
            """
            for i in range(self.num_critics):
                target_state_dict = self.agent.target_critic_states[i].state_dict()
                critic_state_dict = self.agent.critic_states[i].state_dict()
                for key in critic_state_dict:
                    target_state_dict[key] = critic_state_dict[
                        key
                    ] * self.tau + target_state_dict[key] * (1 - self.tau)
                self.agent.target_critic_states[i].load_state_dict(target_state_dict)

        def calc_actor_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs, acts, rews, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )

            # Q-values
            action_distribution = self.agent.get_action_distribution(obs)
            actions = action_distribution.sample()
            q_values = self.agent.get_q_values(obs, actions)
            q_values = torch.mean(q_values)

            # Entropy regularization
            entropy = self.agent.get_entropy(action_distribution)
            entropy = torch.mean(entropy)

            alpha = self.agent.alpha(
                torch.tensor(1.0, device=q_values.device).reshape(1, 1)
            )

            # Maximize loss
            loss = q_values + alpha * entropy
            # Equivalent to minimizing -loss with gradient descent
            loss = -loss

            info = {
                "entropy": entropy,
                "actor_loss": loss,
            }
            return loss, info

        def update_actor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            loss, info = calc_actor_loss(batch)
            # TODO: optimizer
            return info

        def calc_alpha_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs = batch["observations"]

            action_dist: D.Distribution = self.agent.get_action_distribution(obs)
            entropy = self.agent.get_entropy(action_dist)
            entropy = torch.mean(entropy)

            alpha = self.agent.alpha(
                torch.tensor(1.0, device=entropy.device).reshape(1, 1)
            )
            loss = torch.mean(alpha * (entropy - self.target_entropy))

            return loss, {"alpha": alpha, "alpha_loss": loss}

        def update_alpha(agent: SoftActorCritic, batch: dict[str, np.ndarray]):
            loss, info = calc_alpha_loss(batch)

            # TODO: optimizer
            return info

        def train_step(agent, batch):
            metrics = {"loss": 0.0}
            return agent, metrics

        def eval_step(agent, batch):
            return {"loss": 0.0}

        def update_step(batch: dict[str, np.ndarray]):
            # for loop
            for _ in range(self.num_critic_updates):
                info = update_crtics(batch)
                update_target_crtics()
            actor_info = update_actor(batch)
            alpha_info = update_alpha(batch)

            info.update(actor_info)
            info.update(alpha_info)
            # info.update(
            #     {
            #         "actor_lr": self.agent.actor_state.opt_state.hyperparams[
            #             "learning_rate"
            #         ],
            #         "critic_lr": self.agent.critic_states[0].opt_state.hyperparams[
            #             "learning_rate"
            #         ],
            #         "alpha_lr": self.agent.alpha_state.opt_state.hyperparams[
            #             "learning_rate"
            #         ],
            #     }
            # )
            return info

        return train_step, eval_step, update_step
