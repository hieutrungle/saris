import os
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import orbax.checkpoint as ocp
from saris.utils.logger import TensorboardLogger
from saris.drl.infrastructure.train_state import TrainState
from saris.drl.agents.actor_critic import ActorCritic
import gymnasium as gym
import argparse
from saris.utils import utils
from saris.drl.infrastructure.replay_buffer import ReplayBuffer
from flax.training import orbax_utils
import re
import matplotlib.pyplot as plt


class ActorCriticTrainer:
    """
    A basic Trainer module for actor critic algorithms.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_shape: Sequence[int],
        actor_class: Callable,
        actor_hparams: Dict[str, Any],
        critic_class: Callable,
        critic_hparams: Dict[str, Any],
        actor_optimizer_hparams: Dict[str, Any],
        critic_optimizer_hparams: Dict[str, Any],
        num_actor_samples: int = 16,
        num_critic_updates: int = 4,
        num_critics: int = 2,
        discount: float = 0.9,
        ema_decay: float = 0.95,
        grad_accum_steps: int = 1,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """
        A basic Trainer module for actor critic algorithms. This sumerizes most common
        training functionalities like logging, model initialization, training loop, etc.

        Args:
            observation_shape: The shape of the observation space.
            action_shape: The shape of the action space.
            actor_class: The class of the actor model that should be trained.
            actor_hparams: A dictionary of all hyperparameters of the actor model.
              Is used as input to the model when created.
            critic_class: The class of the critic model that should be trained.
            critic_hparams: A dictionary of all hyperparameters of the critic model.
              Is used as input to the model when created.
            actor_optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
                Used during initialization of the optimizer.
            critic_optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
                Used during initialization of the optimizer.
            discount: The discount factor for the environment.
            ema_decay: The decay factor for the target networks.
            grad_accum_steps: The number of steps to accumulate gradients before applying
            seed: Seed to initialize PRNG.
            logger_params: A dictionary containing the specification of the logger.
            enable_progress_bar: If False, no progress bar is shown.
            debug: If True, no jitting is applied. Can be helpful for debugging.
        """

        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.actor_class = actor_class
        self.actor_hparams = actor_hparams
        self.critic_class = critic_class
        self.critic_hparams = critic_hparams
        self.actor_optimizer_hparams = actor_optimizer_hparams
        self.critic_optimizer_hparams = critic_optimizer_hparams
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.num_critics = num_critics
        self.discount = discount
        self.ema_decay = ema_decay
        self.grad_accum_steps = grad_accum_steps
        self.seed = seed
        self.logger_params = logger_params
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        jax.config.update("jax_debug_nans", True)

        # Set of hyperparameters to save
        self.config = {
            "actor_class": actor_class.__name__,
            "actor_hparams": actor_hparams,
            "critic_class": critic_class.__name__,
            "critic_hparams": critic_hparams,
            "actor_optimizer_hparams": actor_optimizer_hparams,
            "critic_optimizer_hparams": critic_optimizer_hparams,
            "num_critics": num_critics,
            "discount": discount,
            "ema_decay": ema_decay,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "grad_accum_steps": grad_accum_steps,
            "seed": self.seed,
        }
        self.config.update(kwargs)

        self.rng_key = random.PRNGKey(self.seed)

        # Create empty actor. Note: no parameters yet
        actor = self.create_model(self.actor_class, self.actor_hparams)
        exmp_inputs = np.zeros(self.observation_shape, dtype=np.float32)
        exmp_inputs = self.add_batch_dimension(exmp_inputs)
        exmp_inputs = (
            [exmp_inputs] if not isinstance(exmp_inputs, (list, tuple)) else exmp_inputs
        )
        # self.print_tabulate(actor, exmp_inputs)
        self.rng_key, actor_rng = random.split(self.rng_key)
        actor_state = self.init_model(actor, exmp_inputs, actor_rng)

        # Create empty critic. Note: no parameters yet
        critic = self.create_model(self.critic_class, self.critic_hparams)
        exmp_inputs = [
            np.zeros(self.observation_shape, dtype=np.float32),
            np.zeros(self.action_shape, dtype=np.float32),
        ]
        exmp_inputs = self.add_batch_dimension(exmp_inputs)
        exmp_inputs = (
            [exmp_inputs] if not isinstance(exmp_inputs, (list, tuple)) else exmp_inputs
        )
        # self.print_tabulate(critic, exmp_inputs)
        critic_states = []
        target_critic_states = []
        for i in range(self.num_critics):
            self.rng_key, critic_rng = random.split(self.rng_key)
            critic_states.append(self.init_model(critic, exmp_inputs, critic_rng))
            target_critic_states.append(copy(critic_states[i]))

        # Create actor critic agent
        self.agent = ActorCritic(
            actor_state=actor_state,
            critic_states=critic_states,
            target_critic_states=target_critic_states,
        )

        # Init trainer parts
        self.logger = self.init_logger(logger_params)
        # self.create_jitted_functions()
        self.checkpoint_manager = self.init_checkpoint_manager(self.logger.log_dir)

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
            print("Creating model with create method")
            model = create_fn(**model_hparams)
        else:
            print("Creating model with init method")
            model = model_class(**model_hparams)
        return model

    def print_tabulate(self, model: nn.Module, exmp_input: Any):
        """
        Prints a summary of the Module represented as table.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        print(model.tabulate(random.PRNGKey(self.seed), *exmp_input, train=True))

    def init_model(self, model: nn.Module, exmp_input: Any, rng: jax.random.PRNGKey):
        """
        Creates an initial training state with newly generated network parameters.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng, init_rng = random.split(rng)

        # Run model initialization
        variables = self.run_model_init(model, exmp_input, init_rng)
        # Create default state. Optimizer is initialized later
        return TrainState(
            step=0,
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            rng=model_rng,
            tx=None,
            opt_state=None,
        )

    def run_model_init(self, model: nn.Module, exmp_input: Any, init_rng: Any) -> Dict:
        """
        The model initialization call

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        return model.init(init_rng, *exmp_input, train=True)

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
                json.dump(self.config, f, indent=4)

        return logger

    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step, update_step = self.create_step_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
            self.update_step = update_step
        else:
            # self.train_step = jax.jit(train_step, donate_argnames=("state",))
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)
            self.update_step = jax.jit(update_step, donate_argnames=("agent"))

    @staticmethod
    def create_step_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
    ]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """

        def train_step(state: TrainState, batch: Any):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            metrics = {}
            return metrics

        def update_step(state: TrainState, batch: Any):
            metrics = {}
            return state, metrics

        raise NotImplementedError

    def init_checkpoint_manager(self, checkpoint_path: str):
        """
        Initializes a checkpoint manager for saving and loading model states.

        Args:
          checkpoint_path: Path to the directory where the checkpoints are stored.
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        async_checkpointer = ocp.AsyncCheckpointer(
            ocp.PyTreeCheckpointHandler(), timeout_secs=50
        )
        options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        return ocp.CheckpointManager(
            checkpoint_path,
            checkpointers=async_checkpointer,
            options=options,
        )

    def init_optimizer(
        self,
        state: TrainState,
        optimizer_hparams: Dict[str, Any],
        num_epochs: int,
        num_steps_per_epoch: int,
    ):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy(optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize learning rate scheduler
        # A cosine decay scheduler is used
        lr = hparams.pop("lr", 1e-3)
        num_train_steps = int(num_epochs * num_steps_per_epoch)
        warmup_steps = hparams.pop("warmup_steps", num_train_steps // 5)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr / 50,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=num_train_steps,
            end_value=lr / 5,
        )

        @optax.inject_hyperparams
        def chain_optimizer(learning_rate: float):
            # Clip gradients at max value, and evt. apply weight decay
            transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 1.0))]
            # wd is integrated in adamw
            if opt_class == optax.sgd and "weight_decay" in hparams:
                transf.append(
                    optax.add_decayed_weights(hparams.pop("weight_decay", 0.0))
                )

            return optax.chain(*transf, opt_class(learning_rate, **hparams))

        optimizer = chain_optimizer(learning_rate=lr_schedule)

        # Initialize training state
        state = TrainState.create(
            apply_fn=state.apply_fn,
            params=state.params,
            batch_stats=state.batch_stats,
            tx=optimizer,
            rng=state.rng,
        )
        return state

    def get_agent(self):
        return self.agent

    def init_agent_optimizer(
        self,
        agent: ActorCritic,
        drl_config: Dict[str, Any],
    ) -> ActorCritic:
        """
        Initializes the optimizer for the agent's components:
        - actor
        - critics
        - target critics
        - other components

        Returns:
            agent: The agent with initialized optimizers.
        """
        raise NotImplementedError

    def train_agent(
        self, env: gym.Env, drl_config: Dict[str, Any], args: argparse.Namespace
    ):
        print("\n" + f"*" * 80)
        print(f"Training {self.actor_class.__name__} and {self.critic_class.__name__}")

        self.agent = self.init_agent_optimizer(self.agent, drl_config)

        # Create replay buffer
        local_assets_dir = utils.get_dir(args.source_dir, "local_assets")
        buffer_saved_name = os.path.join("replay_buffer", drl_config["log_string"])
        buffer_saved_dir = utils.get_dir(local_assets_dir, buffer_saved_name)
        replay_buffer = ReplayBuffer(
            drl_config["replay_buffer_capacity"], buffer_saved_dir, seed=args.seed
        )

        # Load model if exists
        is_ckpt_available = False
        if args.resume and self.logger.log_dir is not None:
            subdirs = [
                name
                for name in os.listdir(self.logger.log_dir)
                if re.match(r"^\d+$", name)
            ]
            if subdirs:
                is_ckpt_available = True
        if is_ckpt_available:
            print(f"Loading model from {self.logger.log_dir}")
            self.load_models()
            start_step = self.checkpoint_manager.latest_step()
        else:
            print(f"Training from scratch")
            start_step = 0
        start_step = int(start_step)

        # Training loop
        (observation, info) = env.reset()

        best_return = -np.inf
        t_range = tqdm(
            range(start_step, drl_config["total_steps"]),
            total=drl_config["total_steps"],
            dynamic_ncols=True,
            initial=start_step,
        )
        for step in t_range:
            # accumulate data in replay buffer
            if step < int(drl_config["random_steps"] * 1 / 2):
                env.unwrapped.location_known = True
                observation, info = env.reset()
                action = env.action_space.sample()
            elif step == int(drl_config["random_steps"] * 1 / 2):
                env.unwrapped.location_known = False
                observation, info = env.reset()
                action = env.action_space.sample()
            elif step < drl_config["random_steps"]:
                env.unwrapped.location_known = False
                action = env.action_space.sample()
            else:
                env.unwrapped.location_known = False
                actions = self.get_agent().get_actions(observation.reshape(1, -1))
                jax.block_until_ready(actions)
                actions = np.asarray(actions, dtype=np.float32)
                action = np.squeeze(actions, axis=0)

            try:
                next_observation, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error in step {step}: {e}")
                time.sleep(2)
                continue

            done = terminated or truncated
            done = np.asarray(done, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32)

            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )

            self.logger.log_metrics({"train_reward": reward}, step)
            self.logger.log_metrics({"train_path_gain_dB": info["path_gain_dB"]}, step)
            if done:
                train_return = float(np.mean(info["episode"]["r"]))
                self.logger.log_metrics({"train_return": train_return}, step)
                self.logger.log_metrics({"train_ep_len": info["episode"]["l"]}, step)
                observation, info = env.reset()
            else:
                observation = next_observation

            # Update agent
            if step >= drl_config["training_starts"]:
                jax.block_until_ready(self.agent)
                for jj in range(drl_config["num_train_steps_per_env_step"]):
                    # print(f"sub_step: {jj}")
                    batch = replay_buffer.sample(drl_config["batch_size"])
                    self.agent, update_info = self.update_step(self.agent, batch)
                    jax.block_until_ready(self.agent)
                    jax.block_until_ready(update_info)

                info.update(update_info)

                # Logging
                if step % args.log_interval == 0:
                    self.logger.log_metrics(info, step)
                    self.logger.flush()
                print(f"Replay Buffer size: {len(replay_buffer)}")

                # Evaluation
                if step % drl_config["eval_interval"] == 0:
                    print(f"Step: {step} - Evaluating agent")
                    eval_trajectories = self.eval_trajectories(env, drl_config)
                    return_mean = np.mean([t["return"] for t in eval_trajectories])
                    return_std = np.std([t["return"] for t in eval_trajectories])
                    eval_metrics = {
                        "eval_return_mean": return_mean,
                        "eval_return_std": return_std,
                    }
                    self.save_metrics(f"eval_metrics_{step}", eval_metrics)
                    self.logger.log_metrics(eval_metrics, step)
                    if return_mean > best_return:
                        best_return = return_mean
                        self.save_models(step)
                        best_metrics = eval_metrics
                        best_metrics.update({"step": step})
                        best_metrics.update(info)
                        for k, v in best_metrics.items():
                            if isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
                                best_metrics[k] = float(np.mean(v))
                        self.save_metrics("best_metrics", best_metrics)

                    # Add traj to replay buffer
                    for traj in eval_trajectories:
                        replay_buffer.insert_batch(
                            observations=traj["obs"],
                            actions=traj["acts"],
                            rewards=traj["rews"],
                            next_observations=traj["next_obs"],
                            dones=traj["dones"],
                        )
                    print(f"After Eval - Replay Buffer size: {len(replay_buffer)}")
                t_range.set_postfix(
                    {
                        "actor_loss": f"{info['actor_loss']:.4e}",
                        "critic_loss": f"{info['critic_loss']:.4e}",
                        "alpha": f"{info['alpha']:.4e}",
                    },
                    refresh=True,
                )
        print(f"Replay Buffer")
        print(f" - Size: {len(replay_buffer)}")
        print(f" - Capacity: {replay_buffer.max_size}")
        print(f" - Rewards: {replay_buffer.rewards}")
        print(f" - Dones: {replay_buffer.dones}")

        self.wait_for_checkpoint()
        print(f"Training complete!\n")

    def eval_trajectory(self, env: gym.Env, drl_config: Dict[str, Any]):
        (ob, info) = env.reset()
        obs, acts, rews, next_obs, dones = [], [], [], [], []
        path_gains = []
        for step in range(drl_config["eval_ep_len"]):
            actions = self.get_agent().get_actions(ob.reshape(1, -1))
            jax.block_until_ready(actions)
            action = np.squeeze(np.asarray(actions, dtype=np.float32))
            try:
                next_ob, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"Error in step {step}: {e}")
                time.sleep(2)
                continue

            done = terminated or truncated
            done = np.asarray(done, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)
            obs.append(ob)
            acts.append(action)
            rews.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
            path_gains.append(info["path_gain_dB"])

            if done:
                (ob, info) = env.reset()
                break
            else:
                ob = next_ob

        obs = np.asarray(obs)
        acts = np.asarray(acts)
        rews = np.asarray(rews)
        next_obs = np.asarray(next_obs)
        dones = np.asarray(dones)
        return {
            "obs": obs,
            "acts": acts,
            "rews": rews,
            "next_obs": next_obs,
            "dones": dones,
            "path_gains": path_gains,
            "return": np.sum(rews),
        }

    def eval_trajectories(self, env: gym.Env, drl_config: Dict[str, Any]):
        trajectories = []
        for i in range(drl_config["num_eval_trials"]):
            trajectory = self.eval_trajectory(env, drl_config)
            trajectories.append(trajectory)
        return trajectories

    def eval_agent(
        self, env: gym.Env, drl_config: Dict[str, Any], args: argparse.Namespace
    ):

        print("\n" + f"*" * 80)
        print(
            f"Evaluating {self.actor_class.__name__} and {self.critic_class.__name__}"
        )
        ep_len = drl_config["eval_ep_len"]
        env.unwrapped.location_known = False
        self.load_models()

        max_step = 0
        num_evals = 3
        gains = np.zeros((num_evals, ep_len))
        rewards = np.zeros((num_evals, ep_len))
        saved_metrics = {}
        for i in range(num_evals):
            (ob, info) = env.reset()
            t_range = tqdm(range(0, ep_len), dynamic_ncols=True)
            for step_ in t_range:
                actions = self.get_agent().get_actions(ob.reshape(1, -1))
                actions = np.asarray(actions, dtype=np.float32)
                action = np.squeeze(actions, axis=0)
                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    print(f"Error in step {step}: {e}")
                    time.sleep(2)
                    continue

                done = terminated or truncated

                max_step = max(max_step, step_)

                self.logger.log_scalar(reward, f"eval_reward_{i}", step_)
                self.logger.log_scalar(
                    info["path_gain_dB"], f"eval_path_gain_dB_{i}", step_
                )
                gains[i, step_] = info["path_gain_dB"]
                rewards[i, step_] = reward

                if done:

                    eval_return = float(np.mean(info["episode"]["r"]))
                    eval_ep_len = info["episode"]["l"][0]
                    saved_metrics.update(
                        {
                            f"eval_return_{i}": eval_return,
                            f"eval_ep_len_{i}": eval_ep_len,
                        }
                    )
                    self.logger.log_scalar(eval_return, f"eval_return_{i}", step_)
                    self.logger.log_scalar(eval_ep_len, f"eval_ep_len_{i}", step_)
                    t_range.set_postfix(
                        {
                            "eval_ep_len": eval_ep_len,
                            "eval_return": f"{eval_return:.4e}",
                        },
                        refresh=True,
                    )
                    break
                else:
                    obs = next_obs
        self.save_metrics("eval_metrics", saved_metrics)

        # plot the evaluation results
        max_step = max_step + 1

        # plot the evaluation rewards
        rewards = rewards[:, :max_step]
        rewards_means = np.mean(rewards, axis=0)
        rewards_stds = np.std(rewards, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.plot(rewards_means, label="mean")
        ax.fill_between(
            range(max_step),
            rewards_means - rewards_stds,
            rewards_means + rewards_stds,
            alpha=0.25,
            # label="min-max range",
        )
        ax.plot(rewards_means - rewards_stds, label="min", linestyle="--")
        ax.plot(rewards_means + rewards_stds, label="max", linestyle="--")
        ax.grid()
        ax.legend()
        ax.set_title("Evaluation Rewards")
        ax.set_xlabel("steps")
        ax.set_ylabel("reward")
        # save the plot
        fig_name = f"eval_rewards_{num_evals}_runs.png"
        fig_path = os.path.join(self.logger.log_dir, fig_name)
        plt.savefig(fig_path)

        # plot the evaluation rewards
        gains = gains[:, :max_step]
        gains_means = np.mean(gains, axis=0)
        gains_stds = np.std(gains, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.plot(gains_means, label="mean")
        ax.fill_between(
            range(max_step),
            gains_means - gains_stds,
            gains_means + gains_stds,
            alpha=0.25,
            # label="min-max range",
        )
        ax.plot(gains_means - gains_stds, label="min", linestyle="--")
        ax.plot(gains_means + gains_stds, label="max", linestyle="--")
        ax.grid()
        ax.legend()
        ax.set_title("Evaluation Gains")
        ax.set_xlabel("steps")
        ax.set_ylabel("gains")
        # save the plot
        fig_name = f"eval_gain_{num_evals}_runs.png"
        fig_path = os.path.join(self.logger.log_dir, fig_name)
        plt.savefig(fig_path)

        # log the evaluation results to tensorboard
        for step in range(max_step):
            metrics = {
                "eval_rewards_means": rewards_means[step],
                "eval_rewards_stds": rewards_stds[step],
                "eval_path_gain_dB_mean": gains_means[step],
                "eval_path_gain_dB_std": gains_stds[step],
            }
            for i in range(num_evals):
                metrics[f"eval_reward_{i}"] = rewards[i, step]
                metrics[f"eval_path_gain_dB_{i}"] = gains[i, step]
            self.logger.log_metrics(metrics, step)

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
        self, data: Union[np.ndarray, jnp.ndarray, list, dict, tuple]
    ) -> Union[jnp.ndarray, list, dict, tuple]:
        """
        Adds a batch dimension to the data.

        Args:
          data: The data to add a batch dimension to.

        Returns:
          The data with a batch dimension added.
        """
        if isinstance(data, dict):
            return {k: self.add_batch_dimension(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.add_batch_dimension(v) for v in data]
        elif isinstance(data, (np.ndarray, jnp.ndarray)):
            return jnp.expand_dims(data, axis=0)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

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

    def save_models(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        ckpt = {
            "agent": self.agent,
            "config": self.config,
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(
            step,
            ckpt,
            save_kwargs={"save_args": save_args},
        )

    def wait_for_checkpoint(self):
        """
        Wait for the checkpoint manager to finish writing checkpoints.
        """
        self.checkpoint_manager.wait_until_finished()

    def load_models(self, step: Optional[int] = None):
        """
        Load the agent's parameters from a file.
        """
        if step == None:
            step = self.checkpoint_manager.best_step()
        target = {
            "agent": self.agent,
            "config": self.config,
        }
        state_dict = self.checkpoint_manager.restore(
            step,
            items=target,
        )
        self.agent = state_dict["agent"]

    def linear2dB(self, x):
        return 10 * jnp.log10(x)

    def dB2linear(self, x):
        return jnp.power(10, x / 10)
