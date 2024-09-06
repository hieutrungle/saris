from typing import Any, Tuple, Callable, Dict
from saris.drl.trainers import ac_trainer
from saris.drl.agents.sac import SoftActorCritic
from saris.drl.networks.alpha import Alpha
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import os


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

    def init_gradient_scaler(self):
        if "cuda" in self.device.type:
            self.actor_scaler = torch.cuda.amp.GradScaler()
            self.critic_scaler = torch.cuda.amp.GradScaler()
            self.alpha_scaler = torch.cuda.amp.GradScaler()
        else:
            raise f"Device {self.device.type} not supported."

    def save_models(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        ckpt = {
            "step": step,
            "agent": self.agent.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "alpha_scheduler": self.alpha_scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.logger.log_dir, f"checkpoints.pt"))

    def load_models(self) -> int:
        """
        Load the agent's parameters from a file.
        """
        ckpt = torch.load(os.path.join(self.logger.log_dir, f"checkpoints.pt"))
        self.agent.load_state_dict(ckpt["agent"])
        if ckpt.get("actor_optimizer", None) is not None:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
            self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
            self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
            self.alpha_scheduler.load_state_dict(ckpt["alpha_scheduler"])
        step = ckpt.get("step", 0)
        return step

    def create_step_functions(self):

        def do_q_backup(q_values: torch.Tensor) -> torch.Tensor:
            """
            Handle Q-values from multiple different target critic networks to produce target values.

            # Clip-Q, clip to the minimum of the two critics' predictions.
            Double Q-learning: Use the critic that would have been selected by the current policy.

            Parameters:
                q_values (torch.Tensor): Q-values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FROM the different critics.
            Returns:
                torch.Tensor: Target values of shape (num_critics, batch_size).
                    Leading dimension corresponds to target values FOR the different critics.
            """

            # Clip Double Q-values
            q_values, _ = torch.min(q_values, dim=0, keepdim=True)
            q_values = torch.repeat_interleave(q_values, self.num_critics, dim=0)

            # # Double Q-learning: swap q_values of 2nd critic with q_values of 1st critic
            # q_values = torch.stack((q_values[1], q_values[0]), dim=0)

            return q_values

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

            critic_loss = 0.5 * nn.functional.mse_loss(q_values, target_q_values)

            critic_info = {
                "q_values": torch.mean(q_values),
                "next_q_values": torch.mean(next_q_values),
                "target_q_values": torch.mean(target_q_values),
                "critic_loss": critic_loss,
            }
            return critic_loss, critic_info

        def update_crtics(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, info = calc_critic_loss(batch)
            info.update({"critic_lr": self.critic_scheduler.get_last_lr()[0]})

            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_scaler.scale(loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.agent.critics.parameters(), max_norm=1.0
            )
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()
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
            q_values = do_q_backup(q_values)

            # Entropy regularization
            entropy = self.agent.get_entropy(
                action_distribution, sample_shape=(self.num_actor_samples,)
            )

            alpha = self.agent.alpha(
                torch.tensor(1.0, device=q_values.device).reshape(1, 1)
            )

            q_values = torch.mean(q_values, dtype=torch.float32)
            entropy = torch.mean(entropy, dtype=torch.float32)
            loss = -(q_values + alpha.float() * entropy)

            info = {
                "entropy": entropy,
                "actor_loss": loss,
            }
            return loss, info

        def update_actor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, info = calc_actor_loss(batch)
            info.update({"actor_lr": self.actor_scheduler.get_last_lr()[0]})

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.actor_scaler.scale(loss).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
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
            loss = torch.mean(
                alpha * (entropy - self.target_entropy), dtype=torch.float32
            )

            return loss, {"alpha": alpha, "alpha_loss": loss}

        def update_alpha(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, info = calc_alpha_loss(batch)
            info.update({"alpha_lr": self.alpha_scheduler.get_last_lr()[0]})

            self.alpha_optimizer.zero_grad(set_to_none=True)
            self.alpha_scaler.scale(loss).backward()
            self.alpha_scaler.unscale_(self.alpha_optimizer)
            torch.nn.utils.clip_grad_norm_(self.agent.alpha.parameters(), max_norm=1.0)
            self.alpha_scaler.step(self.alpha_optimizer)
            self.alpha_scaler.update()
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
