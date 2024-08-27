from typing import Any, Tuple, Callable, Dict
import jax
from jax import numpy as jnp
from saris.drl.trainers import ac_trainer
from saris.drl.infrastructure.train_state import TrainState
from saris import distributions as D
from saris.drl.agents.actor_critic import ActorCritic
from saris.drl.agents.sac import SoftActorCritic
from saris.drl.networks.alpha import Alpha
import numpy as np
from flax import struct
import functools
import copy


class SoftActorCriticTrainer(ac_trainer.ActorCriticTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_entropy = -np.prod(self.action_shape)
        self.rng_key, alpha_key = jax.random.split(self.rng_key)
        alpha = Alpha(0.05)
        exmp_inputs = jnp.array(1.0).reshape(1, 1)
        alpha_state = self.init_model(alpha, exmp_inputs, alpha_key)
        # self.agent = ActorCritic(
        #     actor_state=self.agent.actor_state,
        #     critic_states=self.agent.critic_states,
        #     target_critic_states=self.agent.target_critic_states,
        # )
        self.agent = SoftActorCritic(
            actor_state=self.agent.actor_state,
            critic_states=self.agent.critic_states,
            target_critic_states=self.agent.target_critic_states,
            alpha_state=alpha_state,
        )
        self.create_jitted_functions()

    def init_agent_optimizer(
        self,
        agent: SoftActorCritic,
        drl_config: Dict[str, Any],
    ) -> SoftActorCritic:
        """
        Initializes the optimizer for the agent's components:
        - actor
        - critics
        - target critics
        - other components
        """
        # Initialize optimizer for actor and critic
        actor_state = self.init_optimizer(
            agent.actor_state,
            self.actor_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )

        critic_states = []
        for i in range(self.num_critics):
            state = self.init_optimizer(
                agent.critic_states[i],
                self.critic_optimizer_hparams,
                drl_config["total_steps"],
                drl_config["num_train_steps_per_env_step"]
                * drl_config["num_critic_updates"],
            )
            critic_states.append(state)

        agent = agent.replace(
            actor_state=actor_state,
            critic_states=critic_states,
        )

        alpha_optimizer_hparams = copy.deepcopy(self.actor_optimizer_hparams)
        alpha_state = self.init_optimizer(
            agent.alpha_state,
            alpha_optimizer_hparams,
            drl_config["total_steps"],
            drl_config["num_train_steps_per_env_step"],
        )
        agent = agent.replace(
            alpha_state=alpha_state,
        )
        return agent

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

        def do_q_backup(next_qs: jnp.ndarray):
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
            next_qs = jnp.min(next_qs, axis=0)
            return next_qs

        def calc_critic_loss(
            tuple_critic_params: Tuple[struct.PyTreeNode],
            critic_apply_fns: Tuple[Callable],
            agent: SoftActorCritic,
            batch: dict[str, np.ndarray],
        ) -> Tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            obs, actions, rewards, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )

            # next_actions shape: (num_actor_samples, batch_size, action_dim)
            next_action_distribution: D.Distribution = agent.get_action_distribution(
                next_obs, agent.actor_state.params, agent.actor_state.apply_fn
            )
            next_actions = next_action_distribution.sample(
                seed=agent.actor_state.rng, sample_shape=(self.num_actor_samples,)
            )
            next_obs = jnp.repeat(
                jnp.expand_dims(next_obs, axis=0), self.num_actor_samples, axis=0
            )

            # next_q_values shape: (num_critics, num_actor_samples, batch_size)
            tuple_target_critic_params = tuple(
                [critic_state.params for critic_state in agent.target_critic_states]
            )
            target_critic_apply_fns = tuple(
                [critic_state.apply_fn for critic_state in agent.target_critic_states]
            )
            next_q_values = agent.get_q_values(
                tuple_target_critic_params,
                target_critic_apply_fns,
                next_obs,
                next_actions,
            )

            # next_q_values shape: (num_actor_samples, batch_size)
            next_q_values = do_q_backup(next_q_values)
            # next_q_values shape: (batch_size)
            next_q_values = jnp.mean(next_q_values, axis=0)

            # Entropy regularization
            # next_action_entropy shape: (batch_size)
            next_action_entropy = agent.get_entropy(
                next_action_distribution,
                sample_shape=(self.num_actor_samples,),
                key=agent.actor_state.rng,
            )
            alpha = agent.alpha_state.apply_fn(
                {"params": agent.alpha_state.params}, jnp.array(1.0).reshape(1, 1)
            )
            next_q_values = next_q_values + alpha * next_action_entropy

            target_q_values = rewards + self.discount * (1.0 - dones) * next_q_values
            target_q_values = jnp.expand_dims(target_q_values, axis=0)
            target_q_values = jnp.repeat(target_q_values, self.num_critics, axis=0)

            q_values = agent.get_q_values(
                tuple_critic_params, critic_apply_fns, obs, actions
            )

            # Mask out NaN values
            mask = jnp.isnan(q_values) | jnp.isnan(target_q_values)
            q_values = jnp.where(mask, 0, q_values)
            target_q_values = jnp.where(mask, 0, target_q_values)
            crtic_loss = 0.5 * jnp.mean((q_values - target_q_values) ** 2, where=~mask)

            critic_info = {
                "q_values": jnp.mean(q_values),
                "next_q_values": jnp.mean(next_q_values),
                "target_q_values": jnp.mean(target_q_values),
            }
            return crtic_loss, critic_info

        def update_crtics(
            agent: SoftActorCritic, batch: dict[str, np.ndarray]
        ) -> Tuple[SoftActorCritic, dict[str, jnp.ndarray]]:
            tuple_critic_params = tuple([state.params for state in agent.critic_states])
            critic_apply_fns = tuple([state.apply_fn for state in agent.critic_states])
            grad_fn = jax.value_and_grad(calc_critic_loss, has_aux=True)
            (crtic_loss, (critic_info)), grads = grad_fn(
                tuple_critic_params, critic_apply_fns, agent, batch
            )
            critic_info.update({"critic_loss": crtic_loss})

            c_states = []
            for i in range(len(agent.critic_states)):
                new_state = agent.critic_states[i].apply_gradients(grads=grads[i])
                c_states.append(new_state)
            agent = agent.replace(critic_states=c_states)
            return agent, critic_info

        def update_target_crtics(
            agent: SoftActorCritic,
        ) -> Tuple[SoftActorCritic, dict[str, jnp.ndarray]]:
            """
            Update target critics with moving average of current critics.
            """
            target_critic_states = list(agent.target_critic_states)
            for i in range(self.num_critics):
                old = agent.target_critic_states[i].params
                new = agent.critic_states[i].params
                new_target_params = jax.tree.map(
                    lambda x, y: (1 - self.ema_decay) * x + y * self.ema_decay, new, old
                )
                target_critic_states[i] = agent.target_critic_states[i].replace(
                    step=agent.target_critic_states[i].step + 1,
                    params=new_target_params,
                )
            agent = agent.replace(target_critic_states=target_critic_states)
            return agent, {}

        def update_all_critics(
            carry: Tuple[SoftActorCritic, dict[str, jnp.ndarray]],
            idx: int,
            batch: dict[str, np.ndarray],
        ):
            agent, critic_info = carry
            jax.block_until_ready(agent)
            agent, critic_info = update_crtics(agent, batch)
            jax.block_until_ready(agent)
            agent, _ = update_target_crtics(agent)
            jax.block_until_ready(agent)
            carry = (agent, critic_info)
            jax.block_until_ready(carry)

            return carry, None

        def calc_actor_loss(
            actor_params: struct.PyTreeNode,
            actor_apply_fn: Callable,
            agent: SoftActorCritic,
            batch: dict[str, np.ndarray],
        ) -> Tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            obs, actions, rewards, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )

            # Q-values
            action_distribution = agent.get_action_distribution(
                obs, actor_params, actor_apply_fn
            )
            actions = action_distribution.sample(
                seed=agent.actor_state.rng, sample_shape=(self.num_actor_samples,)
            )
            obs = jnp.repeat(
                jnp.expand_dims(obs, axis=0), self.num_actor_samples, axis=0
            )
            q_values = agent.get_q_values(
                tuple([state.params for state in agent.critic_states]),
                tuple([state.apply_fn for state in agent.critic_states]),
                obs,
                actions,
            )
            q_values = jnp.mean(q_values)

            # Entropy regularization
            entropy = agent.get_entropy(
                action_distribution,
                sample_shape=(self.num_actor_samples,),
                key=agent.actor_state.rng,
            )
            entropy = jax.lax.stop_gradient(jnp.mean(entropy))

            # next Q-values
            next_action_distribution: D.Distribution = agent.get_action_distribution(
                next_obs,
                jax.lax.stop_gradient(agent.actor_state.params),
                agent.actor_state.apply_fn,
            )
            next_actions = next_action_distribution.sample(
                seed=agent.actor_state.rng, sample_shape=(self.num_actor_samples,)
            )
            next_obs = jnp.repeat(
                jnp.expand_dims(next_obs, axis=0), self.num_actor_samples, axis=0
            )
            next_q_values = agent.get_q_values(
                tuple([state.params for state in agent.target_critic_states]),
                tuple([state.apply_fn for state in agent.target_critic_states]),
                next_obs,
                next_actions,
            )
            next_q_values = do_q_backup(next_q_values)
            next_q_values = jnp.mean(next_q_values, axis=0)
            target_q_values = rewards + self.discount * (1.0 - dones) * next_q_values
            target_q_values = jnp.mean(target_q_values)

            # Maximize entropy and return with gradient ascent
            alpha = agent.alpha_state.apply_fn(
                {"params": agent.alpha_state.params}, jnp.array(1.0).reshape(1, 1)
            )
            actor_loss = (q_values - target_q_values) + alpha * entropy
            # Equivalent to minimizing -actor_loss with gradient descent
            actor_loss = -actor_loss

            return actor_loss, {"entropy": entropy}

        def update_actor(
            agent: SoftActorCritic, batch: dict[str, np.ndarray]
        ) -> Tuple[SoftActorCritic, dict[str, jnp.ndarray]]:
            grad_fn = jax.value_and_grad(calc_actor_loss, has_aux=True)
            (actor_loss, (actor_info)), grads = grad_fn(
                agent.actor_state.params, agent.actor_state.apply_fn, agent, batch
            )
            actor_info.update({"actor_loss": actor_loss})

            new_actor_state = agent.actor_state.apply_gradients(grads=grads)
            agent = agent.replace(actor_state=new_actor_state)
            return agent, actor_info

        def calc_alpha_loss(
            alpha_params: struct.PyTreeNode,
            alpha_apply_fn: Callable,
            agent: SoftActorCritic,
            batch: dict[str, np.ndarray],
        ) -> Tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            obs = batch["observations"]

            action_distribution: D.Distribution = agent.get_action_distribution(
                obs, agent.actor_state.params, agent.actor_state.apply_fn
            )
            alpha = alpha_apply_fn(
                {"params": alpha_params}, jnp.array(1.0).reshape(1, 1)
            )
            entropy = agent.get_entropy(
                action_distribution,
                sample_shape=(self.num_actor_samples,),
                key=agent.actor_state.rng,
            )
            entropy = jnp.mean(entropy)
            loss = jnp.mean(alpha * (entropy - self.target_entropy))

            return loss, {"alpha": alpha}

        def update_alpha(agent: SoftActorCritic, batch: dict[str, np.ndarray]):
            grad_fn = jax.value_and_grad(calc_alpha_loss, has_aux=True)
            (alpha_loss, info), alpha_grad = grad_fn(
                agent.alpha_state.params, agent.alpha_state.apply_fn, agent, batch
            )
            info.update({"alpha_loss": alpha_loss})

            new_alpha_state = agent.alpha_state.apply_gradients(grads=alpha_grad)
            agent = agent.replace(alpha_state=new_alpha_state)
            return agent, info

        def train_step(agent, batch):
            metrics = {"loss": 0.0}
            return agent, metrics

        def eval_step(agent, batch):
            return {"loss": 0.0}

        def update_step(agent: SoftActorCritic, batch: dict[str, np.ndarray]):

            # Update critics
            (_, info) = jax.eval_shape(update_crtics, agent, batch)
            info = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), info)
            jax.block_until_ready(info)
            critic_update_fn = functools.partial(update_all_critics, batch=batch)
            (agent, info), _ = jax.lax.scan(
                critic_update_fn,
                init=(agent, info),
                xs=jnp.arange(self.num_critic_updates),
                length=self.num_critic_updates,
            )
            jax.block_until_ready(agent)

            agent, actor_info = update_actor(agent, batch)
            jax.block_until_ready(agent)

            agent, alpha_info = update_alpha(agent, batch)
            jax.block_until_ready(agent)
            info.update(actor_info)
            jax.block_until_ready(info)
            info.update(alpha_info)
            jax.block_until_ready(info)
            return agent, info

        return train_step, eval_step, update_step
