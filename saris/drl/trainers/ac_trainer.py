# Standard libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict

# JAX/Flax
# If you run this code on Colab, remember to install flax and optax
# !pip install --quiet --upgrade flax optax
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax

import orbax.checkpoint as ocp

# Logging
# from tensorboardX import SummaryWriter
from saris.utils.logger import TensorboardLogger


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


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
        self.num_critics = num_critics
        self.discount = discount
        self.ema_decay = ema_decay
        self.grad_accum_steps = grad_accum_steps
        self.seed = seed
        self.logger_params = logger_params
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug

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

        key = random.PRNGKey(self.seed)

        # Create empty actor. Note: no parameters yet
        actor = self.create_model(self.actor_class, self.actor_hparams)
        exmp_inputs = np.zeros(self.observation_shape, dtype=np.float32)
        exmp_inputs = self.add_batch_dimension(exmp_inputs)
        exmp_inputs = (
            [exmp_inputs] if not isinstance(exmp_inputs, (list, tuple)) else exmp_inputs
        )
        self.print_tabulate(actor, exmp_inputs)
        key, actor_rng = random.split(key)
        self.actor_state = self.init_model(actor, exmp_inputs, actor_rng)

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
        self.print_tabulate(critic, exmp_inputs)
        self.critic_states = []
        self.target_critic_states = []
        for i in range(self.num_critics):
            key, critic_rng = random.split(key)
            self.critic_states.append(self.init_model(critic, exmp_inputs, critic_rng))
            self.target_critic_states.append(copy(self.critic_states[i]))

        # Init trainer parts
        self.logger = self.init_logger(logger_params)
        self.create_jitted_functions()
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
        self.log_dir = log_dir

        return logger

    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_step_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            # self.train_step = jax.jit(train_step, donate_argnames=("state",))
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    @staticmethod
    def create_step_functions(
        self,
    ) -> Tuple[
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

        raise NotImplementedError

    def init_checkpoint_manager(self, checkpoint_path: str):
        """
        Initializes a checkpoint manager for saving and loading model states.

        Args:
          checkpoint_path: Path to the directory where the checkpoints are stored.
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        return ocp.CheckpointManager(
            checkpoint_path,
            options=options,
        )

    def init_optimizer(
        self,
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
        # A cosine decay scheduler is used, but others are also possible
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
        # optimizer = optax.chain(*transf, opt_class(lr_schedule, **hparams))

        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            batch_stats=self.state.batch_stats,
            tx=optimizer,
            rng=self.state.rng,
        )

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

        def check_for_skipped_keys(key):
            for skip_key in skipped_keys:
                if str(skip_key).lower() in str(key).lower():
                    return True
            return False

        for key, value in self.__dict__.items():
            if not check_for_skipped_keys(key):
                print(f" - {key}: {value}")
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
            return {key: self.add_batch_dimension(value) for key, value in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.add_batch_dimension(value) for value in data]
        elif isinstance(data, (np.ndarray, jnp.ndarray)):
            return jnp.expand_dims(data, axis=0)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def save_models(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(
                {
                    "actor": self.actor_state,
                    "critic": self.critic_states,
                    "target_critic": self.target_critic_states,
                }
            ),
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
        state_dict = self.checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(
                {
                    "actor": self.actor_state,
                    "critic": self.critic_states,
                    "target_critic": self.target_critic_states,
                }
            ),
        )
        self.actor_state = state_dict["actor"]
        self.critic_states = state_dict["critic"]
        self.target_critic_states = state_dict["target_critic"]
        # self.state = TrainState.create(
        #     apply_fn=self.model.apply,
        #     params=state_dict["params"],
        #     batch_stats=state_dict["batch_stats"],
        #     # batch_stats=state_dict["batch_stats"],
        #     # Optimizer will be overwritten when training starts
        #     tx=self.state.tx if self.state.tx else optax.sgd(0.1),
        #     rng=self.state.rng,
        # )
