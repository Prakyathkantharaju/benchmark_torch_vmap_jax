import torch
import jax 
import jax.numpy as jnp
import math
from flax import linen as jnn
import torch.nn as torch_nn
import functools

from utils import benchmark

@jax.jit
def scaled_dot_product_attention_jax(q, k, v):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = jnn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

def scaled_dot_product_attention_torch(q, k, v):
    d_k = q.shape[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = torch_nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention




@functools.partial(jax.jit, static_argnums=(2,))
@benchmark(100)
def jax_attention_map(x, qkv_projection_params, qkv_projection):
    for i in range(10):
        qkv = qkv_projection.apply(qkv_projection_params, x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        jax.vmap(scaled_dot_product_attention_jax, in_axes=(0, 0, 0))(q, k, v)
    


# @benchmark(100)
# def torch_attention_map(x, qkv_projection):
#     for i in range(10):
#         qkv = qkv_projection(x)
#         q, k, v = torch.split(qkv, [512, 512, 512], dim=-1)
#         scaled_dot_product_attention_torch(q, k, v)

@benchmark(100)
def torch_attention_map(x, qkv_projection):
    for i in range(10):
        qkv = qkv_projection(x)
        q, k, v = torch.split(qkv, [512, 512, 512], dim=-1)
        torch.func.vmap(scaled_dot_product_attention_torch, in_dims=(0, 0, 0))(q, k, v)






if __name__ == "__main__":
    # initial the networks, 
    # the embedding dim is 512
    random_key = jax.random.PRNGKey(0)
    qkv_projection = torch_nn.Linear(3*512,3 * 512, bias=False)
    qkv_projection_jax = jnn.Dense(features=3 * 512, use_bias=False)
    qkv_projections_jax_params = qkv_projection_jax.init(random_key, jax.random.normal(random_key, shape=(3 * 512)))

    # initial the input
    x = torch.randn(10, 10, 512 * 3)
    x_jax = jax.random.normal(random_key, shape=(10, 10, 512 * 3))

    # get the attention map
    _, mean_time_jax, std_dev_time_jax, mean_memory_jax, std_dev_memory_jax  = jax_attention_map(x_jax, qkv_projections_jax_params, qkv_projection_jax)
    _, mean_time_torch, std_dev_time_torch, mean_memory_torch, std_dev_memory_torch     = torch_attention_map(x, qkv_projection)

    print(f"JAX: Mean time: {mean_time_jax:.6f} seconds")
    print(f"JAX: Time std dev: {std_dev_time_jax:.6f} seconds")
    print(f"JAX: Mean memory usage: {mean_memory_jax / 1024:.2f} KB")
    print(f"JAX: Memory std dev: {std_dev_memory_jax / 1024:.2f} KB")
    print(f"Torch: Mean time: {mean_time_torch:.6f} seconds")
    print(f"Torch: Time std dev: {std_dev_time_torch:.6f} seconds")
    print(f"Torch: Mean memory usage: {mean_memory_torch / 1024:.2f} KB")
    print(f"Torch: Memory std dev: {std_dev_memory_torch / 1024:.2f} KB")



