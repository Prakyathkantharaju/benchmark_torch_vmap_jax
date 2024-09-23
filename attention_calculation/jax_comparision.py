import jax
import jax.numpy as jnp
import math
from flax import linen as jnn
import pandas as pd
from utils import benchmark

NUMBER_OF_TESTS = 100

@benchmark(NUMBER_OF_TESTS)
@jax.jit
def scaled_dot_product_attention_jax(qkv):
    d_k = qkv.shape[-1] // 3
    q, k, v = jnp.array_split(qkv, 3, axis=-1)
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    # I am not sure if this is the best or fastest softmax implementation in jax
    attention = jnn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

def jax_attention_benchmark(csv_path: str, device: str = 'cpu', warmup: bool = True):
    # Set JAX to use the specified device
    jax.config.update('jax_platform_name', device)
    
    random_key = jax.random.PRNGKey(0)
    qkv_jax = jax.random.normal(random_key, shape=(10, 10, 512 * 3))

    if warmup:
        print("Warming up...")
        _ = scaled_dot_product_attention_jax(qkv_jax)


    _, mean_time_jax, std_dev_time_jax, mean_memory_jax, std_dev_memory_jax, mean_GPU_memory_jax = scaled_dot_product_attention_jax(qkv_jax)

    results_df = pd.DataFrame({
        'Framework': ['JAX'],
        'Mean Time': [mean_time_jax],
        'Std Time': [std_dev_time_jax],
        'Mean Memory': [mean_memory_jax],
        'Std Memory': [std_dev_memory_jax],
        'Mean GPU Memory': [mean_GPU_memory_jax],
    })

    results_df.to_csv(csv_path, index=False)
    print(f"JAX results have been saved to '{csv_path}'")

if __name__ == "__main__":
    csv_output_path = 'jax_attention_benchmark_results.csv'
    jax_attention_benchmark(csv_output_path)