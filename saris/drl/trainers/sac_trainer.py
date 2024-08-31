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
        self.target_entropy = torch.tensor(self.target_entropy, device=self.device)

        alpha = Alpha(0.05)
        self.summarize_model(alpha, [[1, 1]])

        self.agent = SoftActorCritic(
            actor=self.agent.actor,
            critics=self.agent.critics,
            target_critics=self.agent.target_critics,
            alpha=alpha,
        )
        self.agent = self.agent.to(self.device)

    def init_agent_optimizer(self, drl_config: Dict[str, Any]) -> None:
        """
        Initializes the optimizer for the agent's components:
        - actor
        - critics
        - target critics
        - other components
        """
        # Initialize optimizer for actor and critic
        (self.actor_optimizer, self.actor_scheduler) = self.init_optimizer(
            self.agent.actor,
            self.actor_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )
        (self.critic_optimizer, self.critic_scheduler) = self.init_optimizer(
            self.agent.critics,
            self.critic_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"]
            * drl_config["num_critic_updates"],
        )
        (self.alpha_optimizer, self.alpha_scheduler) = self.init_optimizer(
            self.agent.alpha,
            self.actor_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )

    def create_step_functions(self):

        def accumulate_gradients(agent, batch, rng_key):
            batch_size = batch[0].shape[0]
            num_minibatches = self.grad_accum_steps
            minibatch_size = batch_size // self.grad_accum_steps
            rngs = jax.random.split(rng_key, num_minibatches)
            grad_fn = jax.value_and_grad(actor_loss)

            def _minibatch_step(
                minibatch_idx: jax.Array | int,
            ) -> Tuple[struct.PyTreeNode, jtorch.Tensor]:
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
                carry: Tuple[struct.PyTreeNode, jtorch.Tensor],
                minibatch_idx: jax.Array | int,
            ) -> Tuple[Tuple[struct.PyTreeNode, jtorch.Tensor], None]:
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

            # Clip-Q, clip to the minimum of the two critics' predictions.
            Double Q-learning: Use the critic that would have been selected by the current policy.

            Parameters:
                next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FROM the different critics.
            Returns:
                torch.Tensor: Target values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FOR the different critics.
            """

            # Clip Q-values
            next_qs, _ = torch.min(next_qs, dim=0, keepdim=True)
            next_qs = torch.repeat_interleave(next_qs, self.num_critics, dim=0)

            # Double Q-learning: swap q_values of 2nd critic with q_values of 1st critic
            # next_qs = torch.stack((next_qs[1], next_qs[0]), dim=0)

            return next_qs

        def calc_critic_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs, acts, rews, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )
            batch_size = obs.shape[0]

            with torch.no_grad():

                next_act_dist: D.Distribution = self.agent.get_action_distribution(
                    next_obs
                )
                # next_actions shape: (num_actor_samples, batch_size, action_dim)
                next_actions = next_act_dist.sample((self.num_actor_samples,))
                next_obs = torch.repeat_interleave(
                    next_obs.unsqueeze(0), self.num_actor_samples, dim=0
                )

                # next_q_values shape: (num_critics, num_actor_samples, batch_size)
                next_q_values = self.agent.get_target_q_values(next_obs, next_actions)

                # next_q_values shape: (num_critics, num_actor_samples, batch_size)
                next_q_values = do_q_backup(next_q_values)

                # Entropy regularization
                # next_action_entropy shape: (num_actor_samples, batch_size)
                next_action_entropy = self.agent.get_entropy(
                    next_act_dist, sample_shape=(self.num_actor_samples,)
                )
                # next_action_entropy shape: (num_critics, num_actor_samples, batch_size)
                next_action_entropy = torch.repeat_interleave(
                    next_action_entropy.unsqueeze(0), self.num_critics, dim=0
                )

                alpha = self.agent.alpha(
                    torch.tensor(1.0, device=next_action_entropy.device).reshape(1, 1)
                )
                # next_q_values shape: (num_critics, num_actor_samples, batch_size)
                next_q_values = next_q_values + alpha * next_action_entropy

                # shape: (num_critics, batch_size)
                next_q_values = torch.mean(next_q_values, dim=1)
                rews = torch.repeat_interleave(
                    rews.unsqueeze(0), self.num_critics, dim=0
                )
                dones = torch.repeat_interleave(
                    dones.unsqueeze(0), self.num_critics, dim=0
                )
                target_q_values = rews + self.discount * (1.0 - dones) * next_q_values

            q_values = self.agent.get_q_values(obs, acts)

            crtic_loss = 0.5 * torch.mean((q_values - target_q_values) ** 2)

            critic_info = {
                "q_values": torch.mean(q_values),
                "next_q_values": torch.mean(next_q_values),
                "target_q_values": torch.mean(target_q_values),
                "critic_loss": crtic_loss,
            }
            return crtic_loss, critic_info

        def update_crtics(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            loss, info = calc_critic_loss(batch)
            info.update({"critic_lr": self.critic_scheduler.get_last_lr()[0]})

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            self.critic_scheduler.step()
            return info

        def update_target_crtics() -> None:
            """
            Update target critics with moving average of current critics.
            """

            for i in range(self.num_critics):
                target_critic_dict = self.agent.target_critics[i].state_dict()
                critic_dict = self.agent.critics[i].state_dict()
                for key in critic_dict:
                    target_critic_dict[key] = critic_dict[
                        key
                    ] * self.tau + target_critic_dict[key] * (1 - self.tau)
                self.agent.target_critics[i].load_state_dict(target_critic_dict)

        def calc_actor_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs = batch["observations"]

            # Q-values
            action_distribution = self.agent.get_action_distribution(obs)
            actions = action_distribution.sample()
            q_values = self.agent.get_q_values(obs, actions)
            q_values = torch.mean(q_values)

            # Entropy regularization
            entropy = self.agent.get_entropy(
                action_distribution, sample_shape=(self.num_actor_samples,)
            )
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
            info.update({"actor_lr": self.actor_scheduler.get_last_lr()[0]})

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()

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

        def update_alpha(batch: dict[str, torch.Tensor]):

            loss, info = calc_alpha_loss(batch)
            info.update({"alpha_lr": self.alpha_scheduler.get_last_lr()[0]})

            self.alpha_optimizer.zero_grad()
            loss.backward()
            self.alpha_optimizer.step()
            self.alpha_scheduler.step()

            return info

        def train_step(agent, batch):
            metrics = {"loss": 0.0}
            return agent, metrics

        def eval_step(agent, batch):
            return {"loss": 0.0}

        def update_step(batch: dict[str, torch.Tensor]):
            # for loop
            for _ in range(self.num_critic_updates):
                info = update_crtics(batch)
                update_target_crtics()
            actor_info = update_actor(batch)
            alpha_info = update_alpha(batch)

            info.update(actor_info)
            info.update(alpha_info)
            return info

        return train_step, eval_step, update_step
