from typing import Any, Tuple, Callable, Dict
from saris.drl.trainers import ac_trainer
from saris.drl.agents.sac import SoftActorCritic
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import os


class SoftActorCriticTrainer(ac_trainer.ActorCriticTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_shape).to(self.device)
        ).item()
        self.alpha = self.temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.agent = SoftActorCritic(
            actor=self.agent.actor,
            critics=self.agent.critics,
            target_critics=self.agent.target_critics,
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
            self.agent.actor.parameters(),
            self.actor_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )
        (self.critic_optimizer, self.critic_scheduler) = self.init_optimizer(
            self.agent.critics.parameters(),
            self.critic_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"]
            * drl_config["num_critic_updates"],
        )
        (self.alpha_optimizer, self.alpha_scheduler) = self.init_optimizer(
            [self.log_alpha],
            self.actor_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )

    def init_gradient_scaler(self):
        if "cuda" in self.device.type:
            self.actor_scaler = torch.cuda.amp.GradScaler()
            self.critic_scaler = torch.cuda.amp.GradScaler()
        else:
            raise f"Device {self.device.type} not supported."

    def save_models(self, step: int, checkpoint_file: str = f"checkpoints.pt"):
        """
        Save the agent's parameters to a file.
        """
        ckpt = {
            "step": step,
            "agent": self.agent.state_dict(),
            "alpha": self.alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "alpha_scheduler": self.alpha_scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.logger.log_dir, checkpoint_file))

    def load_models(self, checkpoint_file: str = f"checkpoints.pt") -> int:
        """
        Load the agent's parameters from a file.
        """
        ckpt = torch.load(os.path.join(self.logger.log_dir, checkpoint_file))
        self.agent.load_state_dict(ckpt["agent"])
        self.agent = self.agent.to(self.device)

        is_train = self.__dict__.get("actor_optimizer", False)
        if ckpt.get("actor_optimizer", None) is not None and is_train:
            self.alpha = ckpt["alpha"].to(self.device)
            self.log_alpha = torch.log(torch.tensor(self.alpha, requires_grad=True))
            self.log_alpha = self.log_alpha.to(self.device)
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
            self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
            self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
            self.alpha_scheduler.load_state_dict(ckpt["alpha_scheduler"])
        step = ckpt.get("step", 0)
        return step

    def create_step_functions(self):

        def calc_critic_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs = batch["observations"]
            acts = batch["actions"]
            next_obs = batch["next_observations"]
            rews = batch["rewards"].unsqueeze(1)
            dones = batch["dones"].unsqueeze(1)

            with torch.no_grad():
                next_acts, next_log_pi, _ = self.agent.actor.sample(
                    next_obs, self.num_actor_samples
                )
                next_obs = next_obs.unsqueeze(0)
                next_obs = torch.repeat_interleave(
                    next_obs, self.num_actor_samples, dim=0
                )

                next_q_values = self.agent.get_target_q_values(next_obs, next_acts)
                next_q_values = torch.mean(next_q_values, dim=1)
                min_q_values, _ = torch.min(next_q_values, dim=0)
                min_q_values = min_q_values - self.alpha * next_log_pi

                target_q_values = rews + self.discount * (1.0 - dones) * min_q_values
                target_q_values = target_q_values.unsqueeze(0)
                target_q_values = torch.repeat_interleave(
                    target_q_values, self.num_critics, dim=0
                )

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

            with torch.autocast(device_type=self.device.type, dtype=self.train_dtype):
                loss, info = calc_critic_loss(batch)

            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_scaler.scale(loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.agent.critics.parameters(), max_norm=1.0
            )
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()

            info.update({"critic_lr": self.critic_scheduler.get_last_lr()[0]})
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
                    target_critic_dict[key] = (
                        critic_dict[key] * (1 - self.polyak)
                        + target_critic_dict[key] * self.polyak
                    )
                self.agent.target_critics[i].load_state_dict(target_critic_dict)

        def calc_actor_loss(
            batch: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            obs = batch["observations"]

            # Q-values
            acts, log_pi, _ = self.agent.actor.sample(obs, self.num_actor_samples)
            obs = obs.unsqueeze(0)
            obs = torch.repeat_interleave(obs, self.num_actor_samples, dim=0)
            q_values = self.agent.get_q_values(obs, acts)
            q_values = torch.mean(q_values, dim=1)
            min_q_values, _ = torch.min(q_values, dim=0)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            loss = -torch.mean(
                min_q_values - (self.alpha * log_pi), dtype=torch.float32
            )

            info = {"actor_loss": loss}
            return loss, info

        def update_actor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            with torch.autocast(device_type=self.device.type, dtype=self.train_dtype):
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

            _, log_pi, _ = self.agent.actor.sample(obs)
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            return alpha_loss, {"alpha_loss": alpha_loss}

        def update_alpha(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

            with torch.autocast(device_type=self.device.type, dtype=self.train_dtype):
                loss, info = calc_alpha_loss(batch)
            info.update({"alpha_lr": self.alpha_scheduler.get_last_lr()[0]})

            self.alpha_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.alpha_optimizer.step()
            self.alpha_scheduler.step()

            self.alpha = self.log_alpha.exp()
            info.update({"alpha": self.alpha.clone().item()})

            return info

        def train_step(agent, batch):
            metrics = {"loss": 0.0}
            return agent, metrics

        def eval_step(agent, batch):
            return {"loss": 0.0}

        def update_step(batch: dict[str, torch.Tensor]):

            for _ in range(self.num_critic_updates):
                info = update_crtics(batch)
            update_target_crtics()

            actor_info = update_actor(batch)
            alpha_info = update_alpha(batch)

            info.update(actor_info)
            info.update(alpha_info)
            return info

        return train_step, eval_step, update_step
