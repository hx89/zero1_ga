import jax
import jax.numpy as jnp
import optax
from jax import random, grad, value_and_grad, lax
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import functools
from typing import NamedTuple

# Add a simple state structure similar to MaxText
class TrainingState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    step: int

# Add MaxText-style initialization function
def init_initial_state(model_fn, optimizer, config, rng):
    """Initialize training state similar to MaxText style"""
    # Initialize parameters
    params = model_fn(rng)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Create initial state
    state = TrainingState(
        params=params,
        opt_state=opt_state,
        step=0
    )
    
    return state


# Dummy GEMM model: y = x @ W
def forward(params, x):
    # h = x @ params['W1'].astype(jnp.bfloat16)
    # h = jax.nn.relu(h).astype(jnp.bfloat16)
    # return h @ params['W2'].astype(jnp.bfloat16)
    return x @ params['W1'].astype(jnp.bfloat16)

def loss_fn(params, x, y_true):
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y_true) ** 2)

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
def create_train_step(optimizer, grad_accum_steps=1):
    def train_step(params, opt_state, x, y):
        # Split x and y into microbatches
        microbatch_size = x.shape[0] // grad_accum_steps
        print('microbatch_size: ', microbatch_size)
        print('grad_accum_steps: ', grad_accum_steps)
        
        # Create microbatch data
        x_microbatches = x.reshape(grad_accum_steps, microbatch_size, -1)
        y_microbatches = y.reshape(grad_accum_steps, microbatch_size, -1)
        
        def grad_accum_body(carry, microbatch_data):
            grads_accum, loss_accum = carry
            x_mb, y_mb = microbatch_data
            
            loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
            
            if grads_accum is None:
                new_grads_accum = grads
            else:
                new_grads_accum = jax.tree_util.tree_map(lambda a, b: a + b, grads_accum, grads)
            
            new_loss_accum = loss_accum + loss
            return (new_grads_accum, new_loss_accum), None
        
        # Initialize carry
        init_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        init_carry = (init_grads, 0.0)
        
        # Use scan to accumulate gradients
        microbatch_data = (x_microbatches, y_microbatches)
        (grads_accum, loss_accum), _ = lax.scan(grad_accum_body, init_carry, microbatch_data)
        
        # Average gradients and loss
        grads_accum = jax.tree_util.tree_map(lambda a: a / grad_accum_steps, grads_accum)
        loss_accum = loss_accum / grad_accum_steps

        updates, opt_state = optimizer.update(grads_accum, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_accum
    return train_step

def test_gemm_training(sharding_mode="dp"):
    # Config
    # batch_size, in_dim, out_dim = 16, 8, 4
    batch_size, in_dim, out_dim, hidden_dim = 128, 4096, 4096, 4096*8
    learning_rate = 0.1
    steps = 5
    ga = 2

    # Data
    key = random.PRNGKey(0)
    target_params = 0.5
    x = random.normal(key, (batch_size, in_dim), dtype=jnp.bfloat16)
    y = random.normal(key, (batch_size, out_dim), dtype=jnp.bfloat16)
    # y = x * target_params
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)

    # Params
    params = {
        # 'W1': random.normal(key, (in_dim, hidden_dim), dtype=jnp.float32),  
        # 'W2': random.normal(key, (hidden_dim, out_dim), dtype=jnp.float32)
        'W1': random.normal(key, (in_dim, out_dim), dtype=jnp.float32),  
    }

    # Optimizer
    optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8, eps_root=1e-16, weight_decay=0.1, mu_dtype=jnp.float32)
    opt_state = optimizer.init(params)

    # Mesh and sharding
    devices = jax.devices()
    print("devices: ", devices)
    mesh = Mesh(np.array(devices).reshape(-1), axis_names=("dp",))

    if sharding_mode == "dp":
        param_pspec = P(None)  # replicated
        data_pspec = P("dp")          # shard data over batch
    elif sharding_mode == "fsdp":
        param_pspec = P("dp")  # shard weights over first dim
        data_pspec = P("dp")          # shard data over batch
    else:
        raise ValueError(f"Unknown sharding mode: {sharding_mode}")

    # init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training, rng)
    # with nn_partitioning.axis_rules(config.logical_axis_rules):
    #     abstract_state = jax.eval_shape(init_state_partial)
    # state_logical_annotations = nn.get_partition_spec(abstract_state)
    # state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)

    with mesh:
        # pjit version of train_step
        @pjit.pjit
        def step_fn(params, opt_state, x, y):
            return create_train_step(optimizer, grad_accum_steps=ga)(params, opt_state, x, y)

        # Shard inputs and params
        params = jax.device_put(params, jax.sharding.NamedSharding(mesh, param_pspec))
        # opt_state = jax.device_put(opt_state, jax.sharding.NamedSharding(mesh, param_pspec))
        x = jax.device_put(x, jax.sharding.NamedSharding(mesh, data_pspec))
        y = jax.device_put(y, jax.sharding.NamedSharding(mesh, data_pspec))

        for i in range(steps):
            params, opt_state, loss = step_fn(params, opt_state, x, y)
            # Convert loss to float for proper formatting
            loss_float = float(loss)
            print(f"[{sharding_mode}] Step {i}, Loss: {loss_float:.4f}")

if __name__ == "__main__":
    # test_gemm_training("dp")    # Run with data parallel
    test_gemm_training("fsdp")  # Run with fully sharded data parallel
