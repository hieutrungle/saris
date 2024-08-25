from typing import Any, Tuple, Callable
import jax
from jax import numpy as jnp
from saris.drl.trainers import ac_trainer
from saris.drl.infrastructure.train_state import TrainState
from saris import distributions as D
from saris.drl.agents.actor_critic import ActorCritic
import numpy as np

# Type aliases
PyTree = Any


class SoftActorCriticTrainer(ac_trainer.ActorCriticTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_step_functions(self):

        def accumulate_gradients(agent, batch, rng_key):
            batch_size = batch[0].shape[0]
            num_minibatches = self.grad_accum_steps
            minibatch_size = batch_size // self.grad_accum_steps
            rngs = jax.random.split(rng_key, num_minibatches)
            grad_fn = jax.value_and_grad(actor_loss)

            def _minibatch_step(
                minibatch_idx: jax.Array | int,
            ) -> Tuple[PyTree, jnp.ndarray]:
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
                carry: Tuple[PyTree, jnp.ndarray], minibatch_idx: jax.Array | int
            ) -> Tuple[Tuple[PyTree, jnp.ndarray], None]:
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
            tuple_critic_params: Tuple[PyTree],
            critic_apply_fns: Tuple[Callable],
            agent: ActorCritic,
            batch: dict[str, np.ndarray],
        ):
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

            next_q_values = next_q_values + 0.05 * next_action_entropy
            target_q_values = rewards + self.discount * (1.0 - dones) * next_q_values
            target_q_values = jnp.expand_dims(target_q_values, axis=0)
            target_q_values = jnp.repeat(target_q_values, self.num_critics, axis=0)

            q_values = agent.get_q_values(
                tuple_critic_params, critic_apply_fns, obs, actions
            )

            crtic_loss = 0.5 * jnp.mean((q_values - target_q_values) ** 2)

            critic_info = {
                "q_vals": q_values,
                "next_q_vals": next_q_values,
                "target_q_vals": target_q_values,
                "q_values": jnp.mean(q_values),
                "next_q_values": jnp.mean(next_q_values),
                "target_q_values": jnp.mean(target_q_values),
            }
            return crtic_loss, critic_info

        def update_crtics(agent: ActorCritic, batch: dict[str, np.ndarray]):
            tuple_critic_params = tuple(
                [critic_state.params for critic_state in agent.critic_states]
            )
            critic_apply_fns = tuple(
                [critic_state.apply_fn for critic_state in agent.critic_states]
            )
            grad_fn = jax.value_and_grad(calc_critic_loss, has_aux=True)
            (crtic_loss, (critic_info)), grads = grad_fn(
                tuple_critic_params, critic_apply_fns, agent, batch
            )
            critic_info.update({"critic_loss": crtic_loss})

            c_states = []
            for i in range(len(agent.critic_states)):
                new_state = agent.critic_states[i].apply_gradients(grads=grads[i])
                c_states.append(new_state)
            agent = agent.replace(
                agent.actor_state, c_states, agent.target_critic_states
            )
            return agent, critic_info

        def update_target_crtics(
            agent: ActorCritic,
        ):
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
            agent = agent.replace(
                agent.actor_state, agent.critic_states, target_critic_states
            )
            return agent, {}

        def update_actor():
            pass

        def update_alpha():
            pass

        def train_step(agent, batch):
            metrics = {"loss": 0.0}
            return agent, metrics

        def eval_step(agent, batch):
            return {"loss": 0.0}

        def update_step(agent, batch):
            obs, actions, rewards, next_obs, dones = (
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
            )
            agent, critic_info = update_crtics(agent, batch)
            agent, _ = update_target_crtics(agent)
            metrics = {}
            metrics.update(critic_info)
            return agent, metrics

        return train_step, eval_step, update_step
