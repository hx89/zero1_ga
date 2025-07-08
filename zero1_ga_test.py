import jax
import jax.numpy as jnp
import optax
from jax import random, grad, value_and_grad, lax
from jax.experimental import pjit
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

# Dummy GEMM model: y = x @ W
def forward(params, x):
    # h = x @ params['W1']
    # h = jax.nn.relu(h)
    # return h @ params['W2']
    return x @ params['W1'].astype(jnp.bfloat16)

def loss_fn(params, x, y_true):
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y_true) ** 2)

def create_train_step(optimizer):
    def train_step(params, opt_state, x, y):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss
    return train_step

def test_gemm_training(sharding_mode="dp"):
    # Config
    # batch_size, in_dim, out_dim = 16, 8, 4
    batch_size, in_dim, out_dim, hidden_dim = 16, 4096, 4096, 4096*8
    learning_rate = 0.1
    steps = 5

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

    with mesh:
        # pjit version of train_step
        @pjit.pjit
        def step_fn(params, opt_state, x, y):
            return create_train_step(optimizer)(params, opt_state, x, y)

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
    test_gemm_training("dp")    # Run with data parallel
    # test_gemm_training("fsdp")  # Run with fully sharded data parallel
