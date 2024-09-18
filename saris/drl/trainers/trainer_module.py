import os
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
import copy
from saris.utils.logger import TensorboardLogger
from saris.drl.agents.actor_critic import ActorCritic
import gymnasium as gym
import argparse
from saris.utils import utils, pytorch_utils
from saris.drl.infrastructure.replay_buffer import ReplayBuffer
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary
from torch import optim
import glob
from abc import ABCMeta, abstractmethod
import time


class TrainerModule:
    """
    A basic Trainer module for actor critic algorithms.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_shape: Sequence[int],
        agent_class: Callable,
        agent_hparams: Dict[str, Any],
        agent_optimizer_hparams: Dict[str, Any],
        seed: int,
        logger_params: Dict[str, Any],
        args: argparse.Namespace,
        device: torch.device = torch.device("cpu"),
        train_dtype: torch.dtype = torch.float16,
        enable_progress_bar: bool = True,
        **kwargs,
    ) -> None:
        """
        A basic Trainer module for actor critic algorithms. This sumerizes most common
        training functionalities like logging, model initialization, training loop, etc.

        Args:
            observation_shape: The shape of the observation space.
            action_shape: The shape of the action space.
            agent_class: The class of the agent model that should be trained.
            agent_hparams: A dictionary of all hyperparameters of the agent model.
              Is used as input to the model when created.
            critic_class: The class of the critic model that should be trained.
            critic_hparams: A dictionary of all hyperparameters of the critic model.
              Is used as input to the model when created.
            agent_optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
                Used during initialization of the optimizer.
            seed: Seed to initialize PRNG.
            logger_params: A dictionary containing the specification of the logger.
            enable_progress_bar: If False, no progress bar is shown.
            debug: If True, no jitting is applied. Can be helpful for debugging.
        """

        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.agent_class = agent_class
        self.agent_hparams = agent_hparams
        self.agent_optimizer_hparams = agent_optimizer_hparams
        self.seed = seed
        self.logger_params = logger_params
        self.enable_progress_bar = enable_progress_bar
        self.device = device
        self.train_dtype = train_dtype

        # Set of hyperparameters to save
        self.config = {
            "agent_class": agent_class.__name__,
            "agent_hparams": agent_hparams,
            "agent_optimizer_hparams": agent_optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "seed": self.seed,
        }
        for k, v in args.__dict__.items():
            if v is not None and not isinstance(v, torch.device):
                self.config.update({k: v})

        self.config.update(kwargs)

        self.agent = self.create_model(self.agent_class, self.agent_hparams)

        # Init trainer parts
        self.logger = self.init_logger(logger_params)

    def create_model(self, model_class: Callable, model_hparams: Dict[str, Any]):
        """
        Create a model from a given class and hyperparameters.

        Args:
            model_class: The class of the model that should be created.
            model_hparams: A dictionary of all hyperparameters of the model.
              Is used as input to the model when created.

        Returns:
            model: The created model.
        """
        create_fn = getattr(model_class, "create", None)
        model: nn.Module = None
        if callable(create_fn):
            model = create_fn(**model_hparams)
        else:
            model = model_class(**model_hparams)
        return model

    def summarize_model(self, model: nn.Module, input_shapes: Sequence[Sequence[int]]):
        """
        Prints a summary of the Module represented as table.

        Args:
          input_shapes: A list of input shapes to the model.
        """
        print(f"\nModel: {model.__class__.__name__}")
        summary(
            model,
            input_size=[*input_shapes],
            col_names=["input_size", "output_size", "num_params"],
        )
        print()

    def init_logger(self, logger_params: Optional[Dict] = None):
        """
        Initializes a logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get("log_dir", None)
        if not log_dir:
            base_log_dir = logger_params.get("base_log_dir", "checkpoints/")
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            version = None
        else:
            version = ""

        # Create logger object
        if "log_name" in logger_params:
            log_dir = os.path.join(log_dir, logger_params["log_name"])
        logger_type = logger_params.get("logger_type", "TensorBoard").lower()
        if logger_type == "tensorboard":
            logger = TensorboardLogger(log_dir=log_dir, comment=version)
        elif logger_type == "wandb":
            logger = WandbLogger(
                name=logger_params.get("project_name", None),
                save_dir=log_dir,
                version=version,
                config=self.config,
            )
        else:
            assert False, f'Unknown logger type "{logger_type}"'

        # Save config hyperparameters
        log_dir = logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            with open(os.path.join(log_dir, "hparams.json"), "w") as f:
                json.dump(self.config, f, indent=4, cls=utils.NpEncoder)

        return logger

    @abstractmethod
    def update_step():
        raise NotImplementedError

    def init_optimizer(
        self,
        params: Sequence[nn.Parameter],
        optimizer_hparams: Dict[str, Any],
        num_epochs: int,
        num_steps_per_epoch: int,
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
            params: The parameters of the model to optimize.
            optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            num_epochs: Number of epochs the model will be trained for.
            num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy.copy(optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optim.Adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optim.AdamW
        elif optimizer_name.lower() == "sgd":
            opt_class = optim.SGD
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize learning rate scheduler
        # A cosine decay scheduler is used
        lr = hparams.pop("lr", 1e-3)
        num_train_steps = int(num_epochs * num_steps_per_epoch)
        warmup_steps = hparams.pop("warmup_steps", num_train_steps // 5)
        optimizer = opt_class(params, lr=lr, **hparams)

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1 / 20, total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, num_train_steps - warmup_steps, eta_min=lr / 10
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
        )

        return optimizer, scheduler

    def get_agent(self):
        return self.agent

    @abstractmethod
    def init_agent_optimizer(self, args: argparse.Namespace) -> None:
        """
        Initializes the optimizer for the agent
        """
        (self.agent_optimizer, self.agent_scheduler) = self.init_optimizer(
            self.agent.parameters(),
            self.agent_optimizer_hparams,
            args.num_iterations,
            args.update_epochs,
        )

    @abstractmethod
    def init_gradient_scaler(self):
        """
        Initializes the gradient scaler for mixed precision training.
        """
        if "cuda" in self.device.type:
            self.agent_scaler = torch.cuda.amp.GradScaler()
        else:
            raise f"Device {self.device.type} not supported."

    def train_agent(self, envs: gym.Env, args: argparse.Namespace):
        print("\n" + f"*" * 80)
        print(f"Training {self.agent_class.__name__}")

        self.init_agent_optimizer(args)
        self.init_gradient_scaler()
        self.agent = self.agent.to(self.device)

        # Create replay buffer
        local_assets_dir = utils.get_dir(args.source_dir, "local_assets")
        buffer_saved_name = os.path.join("replay_buffer", args.log_string)
        buffer_saved_dir = utils.get_dir(local_assets_dir, buffer_saved_name)
        # replay_buffer = ReplayBuffer(
        #     args.replay_buffer_capacity, buffer_saved_dir, seed=args.seed
        # )

        # Load model if exists
        is_ckpt_available = False
        if args.resume and self.logger.log_dir is not None:
            ckpt_file = glob.glob(os.path.join(self.logger.log_dir, "*.pt"))
            if ckpt_file:
                is_ckpt_available = True
        if is_ckpt_available:
            print(f"Loading model from {self.logger.log_dir}")
            start_step = self.load_models()
            start_step += 1
        else:
            print(f"Training from scratch")
            start_step = 0
        start_step = int(start_step)

        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(args.device)
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(args.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
        next_obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        )

        global_step = 0
        start_time = time.time()
        next_ob, info = envs.reset(seed=args.seed)
        next_ob = pytorch_utils.from_numpy(next_ob, args.device)
        next_done = torch.zeros(args.num_envs).to(args.device)

        iter_range = tqdm(
            range(start_step, args.num_iterations),
            total=args.num_iterations,
            initial=start_step,
        )

        for iter in iter_range:

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_ob
                dones[step] = next_done

                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(
                        next_ob
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = log_prob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_ob, reward, terminations, truncations, infos = envs.step(
                    pytorch_utils.to_numpy(action)
                )
                next_obs[step] = next_ob
                done = np.logical_or(terminations, truncations)
                rewards[step] = pytorch_utils.from_numpy(reward, args.device).view(-1)
                next_ob = pytorch_utils.from_numpy(next_ob, args.device)
                next_done = torch.Tensor(done).to(args.device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            self.logger.log_metrics(
                                {"charts/episodic_return": info["episode"]["r"]},
                                global_step,
                            )
                            self.logger.log_metrics(
                                {"charts/episodic_length": info["episode"]["l"]},
                                global_step,
                            )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_ob).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(args.device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            utils.save_data(
                {f"{iter}": pytorch_utils.to_numpy(obs)},
                os.path.join(buffer_saved_dir, "obs.txt"),
            )
            utils.save_data(
                {f"{iter}": pytorch_utils.to_numpy(actions)},
                os.path.join(buffer_saved_dir, "actions.txt"),
            )
            utils.save_data(
                {f"{iter}": next_obs.cpu().numpy()},
                os.path.join(buffer_saved_dir, "next_obs.txt"),
            )
            utils.save_data(
                {f"{iter}": pytorch_utils.to_numpy(logprobs)},
                os.path.join(buffer_saved_dir, "rewards.txt"),
            )
            utils.save_data(
                {f"{iter}": pytorch_utils.to_numpy(dones)},
                os.path.join(buffer_saved_dir, "dones.txt"),
            )

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    self.agent_optimizer.zero_grad(set_to_none=True)
                    self.agent_scaler.scale(loss).backward()
                    self.agent_scaler.unscale_(self.agent_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.parameters(), max_norm=0.5
                    )
                    self.agent_scaler.step(self.agent_optimizer)
                    self.agent_scaler.update()
                    self.agent_scheduler.step()

                    # self.agent_optimizer.zero_grad(set_to_none=True)
                    # loss.backward()
                    # nn.utils.clip_grad_norm_(
                    #     self.agent.parameters(), args.max_grad_norm
                    # )
                    # self.agent_optimizer.step()
                    # self.agent_scheduler.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            metrics = {
                "charts/learning_rate": self.agent_optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
            }
            self.logger.log_metrics(metrics, global_step)
            print("SPS:", global_step / (time.time() - start_time))

            iter_range.set_postfix(
                {
                    "policy_loss": f"{pg_loss.item():.4e}",
                    "value_loss": f"{v_loss.item():.4e}",
                    "entropy": f"{entropy_loss.item():.4e}",
                },
                refresh=True,
            )

            if iter % args.save_interval == 0:
                self.save_models(global_step, f"checkpoints_{global_step}.pt")

        print(f"Training complete!\n")
        self.on_training_end()

    def eval_trajectory(self, env: gym.Env, eval_ep_len=30):
        (ob, info) = env.reset()
        ob_shaoe = env.observation_space.shape
        action_shape = env.action_space.shape
        obs = np.full((eval_ep_len, *ob_shaoe), np.nan)
        acts = np.full((eval_ep_len, *action_shape), np.nan)
        rews = np.full(eval_ep_len, np.nan)
        next_obs = np.full((eval_ep_len, *ob_shaoe), np.nan)
        dones = np.full(eval_ep_len, np.nan)
        path_gains = np.full(eval_ep_len, np.nan)

        for step in range(eval_ep_len):
            env.unwrapped.location_known = False
            observations = np.expand_dims(ob, axis=0)
            observations = pytorch_utils.from_numpy(observations, self.device)
            actions = self.agent.get_actions(observations)
            actions = pytorch_utils.to_numpy(actions)
            action = np.squeeze(actions, axis=0)
            try:
                next_ob, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error in step {step}: {e}")
                time.sleep(2)
                continue

            done = terminated or truncated
            done = np.asarray(done, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            obs[step] = ob
            acts[step] = action
            rews[step] = reward
            next_obs[step] = next_ob
            dones[step] = done
            path_gains[step] = info["path_gain_dB"]

            if done:
                (ob, info) = env.reset()
                break
            else:
                ob = next_ob

        return {
            "obs": obs,
            "acts": acts,
            "rews": rews,
            "next_obs": next_obs,
            "dones": dones,
            "path_gains": path_gains,
            "return": np.nansum(rews),
        }

    def eval_trajectories(
        self, env: gym.Env, num_evals: int = 3, eval_ep_len: int = 30
    ) -> Sequence[Dict[str, np.ndarray]]:
        trajectories = []
        env.unwrapped.eval = True
        for i in range(num_evals):
            trajectory = self.eval_trajectory(env, eval_ep_len)
            trajectories.append(trajectory)
        env.unwrapped.eval = False

        return trajectories

    def eval_agent(
        self, env: gym.Env, drl_config: Dict[str, Any], args: argparse.Namespace
    ):

        print("\n" + f"*" * 80)
        print(
            f"Evaluating {self.actor_class.__name__} and {self.critic_class.__name__}"
        )

        eval_ep_len = drl_config["eval_ep_len"]
        num_evals = drl_config["num_eval_trials"]
        env.unwrapped.location_known = False
        env.unwrapped.use_cmap = True
        self.load_models()

        gains = np.full((num_evals, eval_ep_len), np.nan)
        rewards = np.full((num_evals, eval_ep_len), np.nan)
        saved_metrics = {}

        trajectories = self.eval_trajectories(env, num_evals, eval_ep_len)
        for i, traj in enumerate(trajectories):
            eval_return = traj["return"]
            saved_metrics.update(
                {
                    f"eval_return_{i}": eval_return,
                    f"eval_ep_len_{i}": eval_ep_len,
                }
            )
            self.logger.log_scalar(eval_return, f"eval_return_{i}", 0)
            self.logger.log_scalar(eval_ep_len, f"eval_ep_len_{i}", 0)

            for step in range(eval_ep_len):
                gains[i, step] = traj["path_gains"][step]
                rewards[i, step] = traj["rews"][step]
                self.logger.log_scalar(rewards[i, step], f"eval_reward_{i}", step)
                self.logger.log_scalar(gains[i, step], f"eval_path_gain_dB_{i}", step)

        self.save_metrics("final_eval_metrics", saved_metrics)

        rewards_means = np.nanmean(rewards, axis=0)
        rewards_stds = np.nanstd(rewards, axis=0)
        gains_means = np.nanmean(gains, axis=0)
        gains_stds = np.nanstd(gains, axis=0)

        for step in range(eval_ep_len):
            metrics = {
                "eval_rewards_means": rewards_means[step],
                "eval_rewards_stds": rewards_stds[step],
                "eval_path_gain_dB_mean": gains_means[step],
                "eval_path_gain_dB_std": gains_stds[step],
            }
            self.logger.log_metrics(metrics, step)

        # plot the evaluation rewards
        fig_name = f"eval_rewards_{num_evals}_runs.png"
        self.save_fig(rewards_means, rewards_stds, "Evaluation Rewards", fig_name)

        # plot the evaluation path gain
        fig_name = f"eval_gain_{num_evals}_runs.png"
        self.save_fig(gains_means, gains_stds, "Evaluation Gains", fig_name)

    def save_fig(self, means, stds, label, fig_name):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.plot(means, label="mean")
        ax.fill_between(
            range(len(means)),
            means - stds,
            means + stds,
            alpha=0.25,
        )
        ax.plot(means - stds, label="min", linestyle="--")
        ax.plot(means + stds, label="max", linestyle="--")
        ax.grid()
        ax.legend()
        ax.set_title(label)
        ax.set_xlabel("steps")
        ax.set_ylabel(label)
        fig_path = os.path.join(self.logger.log_dir, fig_name)
        plt.savefig(fig_path)

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wraps an iterator in a progress bar tracker (tqdm) if the progress bar
        is enabled.

        Args:
          iterator: Iterator to wrap in tqdm.
          kwargs: Additional arguments to tqdm.

        Returns:
          Wrapped iterator if progress bar is enabled, otherwise same iterator
          as input.
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def print_class_variables(self):
        """
        Prints all class variables of the TrainerModule.
        """
        print()
        print(f"*" * 80)
        print(f"Class variables of {self.__class__.__name__}:")
        skipped_keys = ["state", "variables", "encoder", "decoder"]

        def check_for_skipped_keys(k):
            for skip_key in skipped_keys:
                if str(skip_key).lower() in str(k).lower():
                    return True
            return False

        for k, v in self.__dict__.items():
            if not check_for_skipped_keys(k):
                print(f" - {k}: {v}")
        print(f"*" * 80)
        print()

    def add_batch_dimension(
        self, data: Union[np.ndarray, list, dict, tuple]
    ) -> np.ndarray:
        """
        Adds a batch dimension to the data.

        Args:
          data: The data to add a batch dimension to.

        Returns:
          The data with a batch dimension added.
        """
        if isinstance(data, dict):
            return {k: self.add_batch_dimension(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return self.add_batch_dimension(np.asarray(data))
        else:
            return np.expand_dims(data, axis=0)

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
        filename: Name of the metrics file without folders and postfix.
        metrics: A dictionary of metrics to save in the file.
        """
        with open(
            os.path.join(self.logger.log_dir, f"metrics/{filename}.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=4, cls=utils.NpEncoder)

    @abstractmethod
    def save_models(self, step: int, checkpoint_file: str = f"checkpoints.pt"):
        """
        Save the agent's parameters to a file.
        """
        ckpt = {
            "step": step,
            "agent": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.logger.log_dir, checkpoint_file))

    @abstractmethod
    def load_models(self, checkpoint_file: str = f"checkpoints.pt") -> int:
        """
        Load the agent's parameters from a file.
        """
        ckpt = torch.load(os.path.join(self.logger.log_dir, checkpoint_file))
        self.agent.load_state_dict(ckpt["agent"])
        step = ckpt.get("step", 0)
        return step

    def on_training_end(self):
        """
        End the training process.
        """
        self.logger.flush()
        self.logger.close()
        self.agent = self.agent.to("cpu")
        print("Training ended.")
        print(f"*" * 80)
        print()
        exit()
