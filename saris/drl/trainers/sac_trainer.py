from typing import Any, Tuple, Callable
import jax
from jax import numpy as jnp
from saris.drl.trainers import ac_trainer

# Type aliases
PyTree = Any


class SoftActorCriticTrainer(ac_trainer.ActorCriticTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_step_functions(self):

        # def actor_loss(
        #     tuple_critic_params: Tuple[PyTree],
        #     critic_apply_fns: Tuple[Callable],
        #     target_critic_states: Tuple[ac_trainer.TrainState],
        #     actor_state: ac_trainer.TrainState,
        #     batch: Tuple[jnp.ndarray],
        #     train: bool,
        #     rng_key: jax.random.PRNGKey,
        # ):
        #     obs, acts, rews, next_obs, dones = batch
        #     # pred = self.model.apply(
        #     #     {"params": actor_params}, x, train=train, rngs={"dropout": rng_key}
        #     # )
        #     # axes = tuple(range(1, len(y.shape)))
        #     # loss = jnp.sum(jnp.mean((pred - y) ** 2, axis=axes))
        #     loss = 0
        #     return loss

        # def accumulate_gradients(
        #     actor_state, critic_states, target_critic_states, batch, rng_key
        # ):
        #     batch_size = batch[0].shape[0]
        #     num_minibatches = self.grad_accum_steps
        #     minibatch_size = batch_size // self.grad_accum_steps
        #     rngs = jax.random.split(rng_key, num_minibatches)
        #     grad_fn = jax.value_and_grad(actor_loss)

        #     def _minibatch_step(
        #         minibatch_idx: jax.Array | int,
        #     ) -> Tuple[PyTree, jnp.ndarray]:
        #         """Determine gradients and metrics for a single minibatch."""
        #         minibatch = jax.tree_map(
        #             lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
        #                 x,
        #                 start_index=minibatch_idx * minibatch_size,
        #                 slice_size=minibatch_size,
        #                 axis=0,
        #             ),
        #             batch,
        #         )
        #         step_loss, step_grads = grad_fn(
        #             actor_state.params,
        #             minibatch,
        #             train=True,
        #             rng_key=rngs[minibatch_idx],
        #         )
        #         return step_loss, step_grads

        #     def _scan_step(
        #         carry: Tuple[PyTree, jnp.ndarray], minibatch_idx: jax.Array | int
        #     ) -> Tuple[Tuple[PyTree, jnp.ndarray], None]:
        #         """Scan step function for looping over minibatches."""
        #         step_loss, step_grads = _minibatch_step(minibatch_idx)
        #         carry = jax.tree_map(jnp.add, carry, (step_loss, step_grads))
        #         return carry, None

        #     # Determine initial shapes for gradients and loss.
        #     loss_shape, grads_shapes = jax.eval_shape(_minibatch_step, 0)
        #     grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
        #     loss = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), loss_shape)

        #     # Loop over minibatches to determine gradients and metrics.
        #     (loss, grads), _ = jax.lax.scan(
        #         _scan_step,
        #         init=(loss, grads),
        #         xs=jnp.arange(num_minibatches),
        #         length=num_minibatches,
        #     )

        #     # Average gradients over minibatches.
        #     grads = jax.tree_map(lambda g: g / num_minibatches, grads)
        #     return loss, grads

        def train_step(actor_state, critic_states, target_critic_states, batch):
            # rng, step_rng = jax.random.split(actor_state.rng)
            # loss, grads = accumulate_gradients(
            #     actor_state, critic_states, target_critic_states, batch, step_rng
            # )
            # actor_state = actor_state.apply_gradients(grads=grads, rng=rng)
            # metrics = {"loss": loss}
            metrics = {"loss": 0.0}
            return actor_state, metrics

        def eval_step(actor_state, critic_states, target_critic_states, batch):
            # loss = actor_loss(
            #     actor_state.params, batch, train=False, rng_key=actor_state.rng
            # )
            return {"loss": 0.0}
            # loss = actor_loss(
            #     actor_state.params, batch, train=False, rng_key=actor_state.rng
            # )
            # return {"loss": loss}

        return train_step, eval_step
