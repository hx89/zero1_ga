import jax
import jax.numpy as jnp
import optax
from jax import random, grad, value_and_grad, lax
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import functools
from typing import NamedTuple
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
# from MaxText.layers import linears

# Add a simple state structure similar to MaxText
class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    step: int

def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
  return state

# Add MaxText-style initialization function
def init_initial_state(model, optimizer, rng):
    """Initialize training state similar to MaxText style"""
    # Initialize parameters using the Flax model
    params = model.init(rng, jnp.zeros((1, model.in_dim), dtype=model.dtype))
    return init_training_state(model.apply, params, optimizer)

def get_input_data_sharding(config, mesh):
  """Get the input data sharding for the model"""
  return nn.logical_to_mesh_sharding(P(*config.input_data_sharding_logical_axes), mesh, config.logical_axis_rules)


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def add_data_to_sharding(mesh, path, aval, sharding):
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise AssertionError(f"Expected NamedSharding, found {sharding} of {type(sharding)=} at {jax.tree_util.keystr(path)}")
  try:
    sharded_shape = sharding.shard_shape(aval.shape)
  except Exception as e:
    raise AssertionError(f"Could not shard value {jax.tree_util.keystr(path)} of shape={aval.shape} with {sharding=}") from e
  pspec = sharding.spec

  if 'data' in jax.tree.leaves(pspec):
    return sharding

  for idx, (size, partition) in enumerate(zip(sharded_shape, pspec)):
    if partition is None:
      partition = ()

    if isinstance(partition, str):
      partition = (partition,)

    if size % mesh.shape['dp'] == 0 and (partition is None or 'tensor' not in partition):
      added_component = ('dp',) + partition
      new_pspec = jax.sharding.PartitionSpec(*(pspec[:idx] + (added_component,) + pspec[idx+1:]))
      new_sharding = jax.sharding.NamedSharding(sharding.mesh, new_pspec)
      # return sharding.with_spec(new_pspec)
      return new_sharding
  return sharding

def maybe_update_params_sharding_with_opt(state_mesh_shardings):
  prev_params_shardings = state_mesh_shardings.params
  if isinstance(state_mesh_shardings.opt_state, optax.ScaleByAdamState):
    sharded_fp32_params = state_mesh_shardings.opt_state.mu
  elif isinstance(state_mesh_shardings.opt_state, tuple) and isinstance(state_mesh_shardings.opt_state[0], optax.ScaleByAdamState):
    sharded_fp32_params = state_mesh_shardings.opt_state[0].mu
  else:
    raise NotImplementedError(f"Could not find optimizer state shardings from optimizer of type {type(state_mesh_shardings.opt_state)}")
  if "params" not in sharded_fp32_params.keys():
    # When quantization=fp8 is enabled the sharded_fp32_params
    # are not wrapped in `params`. Here we wrap them back.
    sharded_fp32_params = {"params": sharded_fp32_params}
  state_mesh_shardings = state_mesh_shardings.replace(params=dict(prev_params_shardings, **sharded_fp32_params))
  return prev_params_shardings, state_mesh_shardings

# Flax Linen module for the model
class SimpleLinearModel(nn.Module):
    """Simple linear model: y = x @ W"""
    in_dim: int
    out_dim: int
    dtype: jnp.dtype = jnp.bfloat16
    weights_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Apply linear transformation
        x = x.astype(self.dtype)
        x = nn.Dense(
            features=self.out_dim,
            use_bias=False,
            # kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("batch", None)),
            kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), (None, None)),
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            name='W1'
        )(x)
        return x

# # Alternative approach: Create a separate model with remat
# class SimpleLinearModelWithRemat(nn.Module):
#     """Simple linear model with remat for gradient checkpointing"""
#     in_dim: int
#     out_dim: int
#     dtype: jnp.dtype = jnp.bfloat16
#     weights_dtype: jnp.dtype = jnp.float32
    
#     @nn.compact
#     def __call__(self, x):
#         # Apply remat to the entire forward pass
#         layer = SimpleLinearModel(in_dim=self.in_dim, out_dim=self.out_dim, dtype=self.dtype, weights_dtype=self.weights_dtype)
#         remat_layer = nn.remat(layer)
#         return remat_layer(x)

def loss_fn(model, params, x):
    """Loss function using the Flax model"""
    y_pred = model.apply(params, x)
    # y_pred = nn.remat(model.apply)(params, x)
    return jnp.mean((y_pred - x) ** 2)

# def create_train_step(optimizer, grad_accum_steps=1):
#     def train_step(params, opt_state, x, y):
#         # Split x and y into microbatches
#         microbatch_size = x.shape[0] // grad_accum_steps
#         grads_accum = None
#         loss_accum = 0.0

#         for i in range(grad_accum_steps):
#             x_mb = x[i * microbatch_size : (i + 1) * microbatch_size]
#             y_mb = y[i * microbatch_size : (i + 1) * microbatch_size]
#             loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
#             if grads_accum is None:
#                 grads_accum = grads
#             else:
#                 grads_accum = jax.tree_util.tree_map(lambda a, b: a + b, grads_accum, grads)
#             loss_accum += loss

#         # Average gradients and loss
#         grads_accum = jax.tree_util.tree_map(lambda a: a / grad_accum_steps, grads_accum)
#         loss_accum = loss_accum / grad_accum_steps

#         updates, opt_state = optimizer.update(grads_accum, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return new_params, opt_state, loss
#     return train_step

# Scan loop version
def create_train_step(optimizer, model, mesh, grad_accum_steps=1, params_shardings=None):
    def train_step(state, x):
        params = state.params
        opt_state = state.opt_state
        # Split x and y into microbatches
        microbatch_size = x.shape[0] // grad_accum_steps
        print('microbatch_size: ', microbatch_size)
        print('grad_accum_steps: ', grad_accum_steps)
        
        # Create microbatch data
        x_microbatches = x.reshape(grad_accum_steps, microbatch_size, -1)
        
        def grad_accum_body(carry, microbatch_data):
            params, grads_accum, loss_accum = carry
            x_mb = microbatch_data
            
            loss, grads = jax.value_and_grad(loss_fn, argnums=1)(model, params, x_mb)
            
            if grads_accum is None:
                new_grads_accum = grads
            else:
                new_grads_accum = jax.tree_util.tree_map(lambda a, b: a + b, grads_accum, grads)

            new_loss_accum = loss_accum + loss
            return (params, new_grads_accum, new_loss_accum), None
        
        # Initialize carry
        # Unshard params to trigger AG
        params = jax.tree.map(jax.lax.with_sharding_constraint, params, params_shardings)
        # params = jax.device_put(params, jax.sharding.NamedSharding(mesh, P(None, None)))
        init_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        init_carry = (params, init_grads, 0.0)
        
        # Use scan to accumulate gradients
        microbatch_data = x_microbatches
        (_, grads_accum, loss_accum), _ = lax.scan(grad_accum_body, init_carry, microbatch_data)
        
        # Unshard grads_accum 
        grads_accum = jax.tree.map(jax.lax.with_sharding_constraint, grads_accum, params_shardings)

        # Average gradients and loss
        grads_accum = jax.tree_util.tree_map(lambda a: a / grad_accum_steps, grads_accum)
        loss_accum = loss_accum / grad_accum_steps

        # updates, opt_state = optimizer.update(grads_accum, opt_state, params)
        # new_params = optax.apply_updates(params, updates)
        # return new_params, opt_state, loss_accum
        new_state = state.apply_gradients(grads=grads_accum)
        return new_state, loss_accum
    return train_step

def test_gemm_training(sharding_mode="dp"):
    # Config
    # batch_size, in_dim, out_dim = 16, 8, 4
    batch_size, in_dim, out_dim, hidden_dim = 128, 4096, 4096, 4096*8
    learning_rate = 0.1
    steps = 5
    ga = 2

    # Create the Flax model
    model = SimpleLinearModel(in_dim=in_dim, out_dim=out_dim, dtype=jnp.bfloat16, weights_dtype=jnp.float32)
    # model = SimpleMaxtextLinearModel(in_dim=in_dim, out_dim=out_dim, dtype=jnp.bfloat16, weights_dtype=jnp.float32)
    # Create config-like object for logical axis rules
    class Config:
        def __init__(self):
            self.logical_axis_rules = [
                ('batch', 'dp'),
                ('embed', None),
                ('mlp', None),
                ('heads', None),
                ('kv', None),
            ]
            self.input_data_sharding_logical_axes = ('batch', 'embed')
    config = Config()

    # Data
    key = random.PRNGKey(0)
    target_params = 0.5
    x = random.normal(key, (batch_size, in_dim), dtype=jnp.bfloat16)
    # y = x * target_params
    print("x shape: ", x.shape)

    # Params
    # params = {
    #     # 'W1': random.normal(key, (in_dim, hidden_dim), dtype=jnp.float32),  
    #     # 'W2': random.normal(key, (hidden_dim, out_dim), dtype=jnp.float32)
    #     'W1': random.normal(key, (in_dim, out_dim), dtype=jnp.float32),  
    # }

    # Optimizer
    optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8, eps_root=1e-16, weight_decay=0.1, mu_dtype=jnp.float32)
    # opt_state = optimizer.init(params)

    # Mesh and sharding
    devices = jax.devices()
    print("devices: ", devices)
    mesh = Mesh(np.array(devices).reshape(-1), axis_names=("dp",))

    if sharding_mode == "dp":
        param_pspec = P(None)  # replicated
        opt_pspec = P(None)
        data_pspec = P("dp")          # shard data over batch
    elif sharding_mode == "fsdp":
        param_pspec = P("dp")  # shard weights over first dim
        opt_pspec = P("dp")
        data_pspec = P("dp")          # shard data over batch
    elif sharding_mode == "zero1":
        param_pspec = P(None)
        opt_pspec = P("dp")
        data_pspec = P("dp")          # shard data over batch
    else:
        raise ValueError(f"Unknown sharding mode: {sharding_mode}")

    # MaxText-style state initialization
    if nn is not None and nn_partitioning is not None:
        print('Using MaxText sharding')
        # Use MaxText's nn module if available
        init_state_partial = functools.partial(init_initial_state, model, optimizer, key)
        with nn_partitioning.axis_rules(config.logical_axis_rules):
            abstract_state = jax.eval_shape(init_state_partial)
        state_logical_annotations = nn.get_partition_spec(abstract_state)
        state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)

        # Create new state_mesh_shardings with data sharding added to opt_state
        state_mesh_shardings_w_data = jax.tree_util.tree_map(lambda x: x, state_mesh_shardings)
        state_mesh_shardings_w_data = state_mesh_shardings_w_data.replace(
            opt_state=jax.tree.map_with_path(
                functools.partial(add_data_to_sharding, mesh), 
                unbox_logicallypartioned(abstract_state).opt_state, 
                state_mesh_shardings_w_data.opt_state
            )
        )
        # Shard params to be the same as the opt_state, keep the orginal params shardings in params_shardings
        params_shardings, state_mesh_shardings_w_data = maybe_update_params_sharding_with_opt(state_mesh_shardings_w_data)
    else:
        print('Using fallback manual sharding')
        # Fallback: create shardings manually
        state_mesh_shardings = {
            'params': jax.sharding.NamedSharding(mesh, param_pspec),
            'opt_state': jax.sharding.NamedSharding(mesh, opt_pspec),
            # 'step': jax.sharding.NamedSharding(mesh, P(None))
        }
    data_sharding = get_input_data_sharding(config, mesh)
    in_shardings = (state_mesh_shardings_w_data, data_sharding)  # State, batch
    out_shardings = (state_mesh_shardings_w_data, None)  # State, metrics
    print('data_sharding: ', data_sharding)

    # Create the train step function
    train_step_fn = create_train_step(optimizer, model, mesh, grad_accum_steps=ga, params_shardings=params_shardings)
    
    # JIT the train step function
    p_train_step = jax.jit(
        train_step_fn, 
        in_shardings=in_shardings, 
        out_shardings=out_shardings, 
        # donate_argnums=(0, 1)  # Donate params and opt_state
    )

    # Initialize parameters and optimizer state
    # state = init_initial_state(model, optimizer, key)
    init_state_partial = functools.partial(init_initial_state, model, optimizer)
    init_state_partial.__name__ = "initialize_state"
    # params_shardings, _state_mesh_shardings = maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

    # pylint: disable=not-callable
    state = jax.jit(
        init_state_partial,
        in_shardings=None,
        out_shardings=state_mesh_shardings_w_data,
    )(key)
    state = unbox_logicallypartioned(state)

    for i in range(steps):
        example_batch = jax.lax.with_sharding_constraint(x, data_sharding)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          # Apply sharding constraint to state to match expected sharding
          state = jax.lax.with_sharding_constraint(state, state_mesh_shardings_w_data)
          state, loss = p_train_step(state, example_batch)
          loss_float = float(loss)
          print(f"[{sharding_mode}] Step {i}, Loss: {loss_float:.4f}")

if __name__ == "__main__":
    # test_gemm_training("dp")    # Run with data parallel
    # test_gemm_training("fsdp")  # Run with fully sharded data parallel
    test_gemm_training("zero1")  # Run with fully sharded data parallel
