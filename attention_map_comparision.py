import torch
import jax 
import jax.numpy as jnp
import math
from flax import linen as jnn
import torch.nn as torch_nn
import functools

from utils import benchmark

@benchmark(100)
@jax.jit
def scaled_dot_product_attention_jax(qkv):
    d_k = qkv.shape[-1] // 3
    q, k, v = jnp.split(qkv, [d_k, d_k, d_k], axis=-1)
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = jnn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

@benchmark(100)
def scaled_dot_product_attention_torch(qkv):
    d_k = qkv.shape[-1] // 3
    q, k, v = torch.split(qkv, [d_k, d_k, d_k], dim=-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = torch_nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention






# @benchmark(100)
# def torch_attention_map(x, qkv_projection):
#     for i in range(10):
#         qkv = qkv_projection(x)
#         q, k, v = torch.split(qkv, [512, 512, 512], dim=-1)
#         scaled_dot_product_attention_torch(q, k, v)







if __name__ == "__main__":
    # initial the networks, 
    # the embedding dim is 512
    random_key = jax.random.PRNGKey(0)

    # initial the input
    qkv_torch = torch.randn(10, 10, 512 * 3)
    qkv_jax = jax.random.normal(random_key, shape=(10, 10, 512 * 3))

    # get the attention map
    _, mean_time_jax, std_dev_time_jax, mean_memory_jax, std_dev_memory_jax  = scaled_dot_product_attention_jax(qkv_jax)
    _, mean_time_torch, std_dev_time_torch, mean_memory_torch, std_dev_memory_torch     = scaled_dot_product_attention_torch(qkv_torch)

    print(f"JAX: Mean time: {mean_time_jax:.6f} seconds")
    print(f"JAX: Time std dev: {std_dev_time_jax:.6f} seconds")
    print(f"JAX: Mean memory usage: {mean_memory_jax / 1024:.2f} KB")
    print(f"JAX: Memory std dev: {std_dev_memory_jax / 1024:.2f} KB")
    print(f"Torch: Mean time: {mean_time_torch:.6f} seconds")
    print(f"Torch: Time std dev: {std_dev_time_torch:.6f} seconds")
    print(f"Torch: Mean memory usage: {mean_memory_torch / 1024:.2f} KB")
    print(f"Torch: Memory std dev: {std_dev_memory_torch / 1024:.2f} KB")



