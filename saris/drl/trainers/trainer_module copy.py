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
from tensorboardX import SummaryWriter


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


class TrainerModule:

    def __init__(
        self,
        model_class: nn.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        exmp_input: Any,
        grad_accum_steps: int = 1,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 10,
        **kwargs,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.grad_accum_steps = grad_accum_steps

        # Set of hyperparameters to save
        self.config = {
            "model_class": model_class.__name__,
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "grad_accum_steps": grad_accum_steps,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
        }
        self.config.update(kwargs)

        # Create empty model. Note: no parameters yet
        create_fn = getattr(self.model_class, "create", None)
        self.model: nn.Module = None
        if callable(create_fn):
            print("Creating model with create method")
            self.model = create_fn(**self.model_hparams)
        else:
            print("Creating model with init method")
            self.model = self.model_class(**self.model_hparams)
        exmp_input = (
            [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
        )
        self.print_tabulate(exmp_input)

        # Init trainer parts
        self.logger = self.init_logger(logger_params)
        self.create_jitted_functions()
        self.state = self.init_model(exmp_input)
        self.checkpoint_manager = self.init_checkpoint_manager(self.logger.logdir)

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
            logger = SummaryWriter(logdir=log_dir, comment=version)
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
        log_dir = logger.logdir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            with open(os.path.join(log_dir, "hparams.json"), "w") as f:
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

        return logger

    def init_model(self, exmp_input: Any):
        """
        Creates an initial training state with newly generated network parameters.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)

        # Run model initialization
        variables = self.run_model_init(exmp_input, init_rng)
        # Create default state. Optimizer is initialized later
        return TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            rng=model_rng,
            tx=None,
            opt_state=None,
        )

    def run_model_init(self, exmp_input: Any, init_rng: Any) -> Dict:
        """
        The model initialization call

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        return self.model.init(init_rng, *exmp_input, train=True)

    def print_tabulate(self, exmp_input: Any):
        """
        Prints a summary of the Module represented as table.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        print(self.model.tabulate(random.PRNGKey(0), *exmp_input, train=True))

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
            self.train_step = jax.jit(train_step, donate_argnames=("state",))
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

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

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

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
    ) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        assert num_epochs > 0, "Number of epochs must be larger than 0"
        assert (
            num_epochs // self.check_val_every_n_epoch > 0
        ), f"Number of epochs ({num_epochs}) must be larger than check_val_every_n_epoch ({self.check_val_every_n_epoch})"

        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))

        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        eval_metrics = {
            "val/loss": 0.0,
        }
        train_metrics = {"train/loss": 0.0}
        t = self.tracker(range(1, num_epochs + 1), desc="Epochs")
        logdir = self.logger.logdir
        # jax.profiler.start_trace(os.path.join(logdir, "jax_trace"))
        for epoch_idx in t:
            # with jax.profiler.StepTraceAnnotation("train_epoch", step_num=epoch_idx):
            train_metrics = self.train_epoch(train_loader, log_prefix="train/")

            self.log_metrics(train_metrics, step=epoch_idx)
            # self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)

            # with jax.profiler.StepTraceAnnotation(
            #     "validation_epoch", step_num=epoch_idx
            # ):
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.log_metrics(eval_metrics, step=epoch_idx)
                # self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)

            t.set_postfix(
                {
                    "train_loss": f"{train_metrics['train/loss']:.4e}",
                    "val_loss": f"{eval_metrics['val/loss']:.4e}",
                },
                refresh=True,
            )
        self.wait_for_checkpoint()
        # jax.profiler.stop_trace()

        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.log_metrics(test_metrics, step=epoch_idx)
            # self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)

        # Close logger
        # self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(
        self, train_loader: Iterator, log_prefix: Optional[str] = "train/"
    ) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.
          log_prefix: Prefix to add to all metrics (e.g. 'train/')

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            # for batch in self.tracker(train_loader, desc="Training", leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics[log_prefix + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        metrics["learning_rate"] = float(
            self.state.opt_state.hyperparams["learning_rate"]
        )
        return metrics

    def eval_model(
        self, data_loader: Iterator, log_prefix: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            for key, value in step_metrics.items():
                metrics[key] += value * batch_size
            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics

    def is_new_model_better(
        self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]
    ) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [
            ("val/val_metric", False),
            ("val/acc", True),
            ("val/loss", False),
        ]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

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

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Logs a dictionary of metrics to the logger.

        Args:
          metrics: A dictionary of metrics to log.
          step: The current training step.
        """
        for k, v in metrics.items():
            self.logger.add_scalar(k, v, step)

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(
        self, epoch_idx: int, eval_metrics: Dict[str, Any], val_loader: Iterator
    ):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        pass

    def save_model(self, step: int):
        """
        Save the agent's parameters to a file.
        """
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(
                {"params": self.state.params, "batch_stats": self.state.batch_stats}
            ),
        )

    def wait_for_checkpoint(self):
        """
        Wait for the checkpoint manager to finish writing checkpoints.
        """
        self.checkpoint_manager.wait_until_finished()

    def load_model(self, step: int = None):
        """
        Load the agent's parameters from a file.
        """
        if step == None:
            step = self.checkpoint_manager.best_step()
        state_dict = self.checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(
                {"params": self.state.params, "batch_stats": self.state.batch_stats}
            ),
        )
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict["params"],
            batch_stats=state_dict["batch_stats"],
            # batch_stats=state_dict["batch_stats"],
            # Optimizer will be overwritten when training starts
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
            rng=self.state.rng,
        )

    def bind_model(self):
        """
        Returns a model with parameters bound to it. Enables an easier inference
        access.

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {"params": self.state.params}
        if self.state.batch_stats:
            params["batch_stats"] = self.state.batch_stats
        return self.model.bind(params)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, exmp_input: Any) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file"
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        hparams.pop("model_class")
        hparams.update(hparams.pop("model_hparams"))
        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(exmp_input=exmp_input, **hparams)
        trainer.load_model()
        return trainer

    def finalize(self):
        """
        Method called at the end of the training. Can be used for final logging
        or similar.
        """
        pass

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
