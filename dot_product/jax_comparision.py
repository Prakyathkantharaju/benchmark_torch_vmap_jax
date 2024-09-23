import jax
import jax.numpy as jnp
from utils import benchmark
import pandas as pd

NUMBER_OF_TESTS = 100
n, m = 10000, 100

@benchmark(NUMBER_OF_TESTS)
def dot_product_simple_jax(a: jax.Array, b: jax.Array):
    result_dot = jnp.empty((m, 1))
    for i in range(m):
        result_dot = result_dot.at[i].set(jnp.dot(a[i], b[i]))
    return result_dot

@benchmark(NUMBER_OF_TESTS)
def dot_product_jax_vmap(a: jax.Array, b: jax.Array):
    return jax.vmap(jnp.dot)(a, b)

@benchmark(NUMBER_OF_TESTS)
# this is for jitted function first jit the function then apply vmap
@jax.jit
def jax_dot_jit(a: jax.Array, b: jax.Array):
    return jax.vmap(jnp.dot)(a, b)

def jax_dot(csv_path: str, device : str = 'cpu', warmup: bool = True):
    # Set JAX to use CPU only
    jax.config.update('jax_platform_name', device)
    
    random_key = jax.random.PRNGKey(0)
    a = jax.random.normal(random_key, shape=(m, n))
    b = jax.random.normal(random_key, shape=(m, n))
    
    # jax_result, jax_mean, jax_std, jax_memory, jax_memory_std, jax_GPU_memory = dot_product_simple_jax(a, b)
    jax_vmap_result, jax_vmap_mean, jax_vmap_std, jax_vmap_memory, jax_vmap_memory_std, jax_vmap_GPU_memory = dot_product_jax_vmap(a, b)
    if warmup:
        print("Warming up...")
        _ = jax_dot_jit(a, b)
    jax_jit_result, jax_jit_mean, jax_jit_std, jax_jit_memory, jax_jit_memory_std, jax_jit_GPU_memory = jax_dot_jit(a, b)
    # Create a DataFrame from the benchmark results
    results_df = pd.DataFrame({
        'Framework': [ 'JAX VMAP', 'JAX JIT'],
        'Mean Time': [jax_vmap_mean, jax_jit_mean],
        'Std Time': [jax_vmap_std, jax_jit_std],
        'Mean Memory': [jax_vmap_memory, jax_jit_memory],
        'Std Memory': [jax_vmap_memory_std, jax_jit_memory_std],
        'Mean GPU Memory': [jax_vmap_GPU_memory, jax_jit_GPU_memory],
    })

    # Save results to CSV
    results_df.to_csv(csv_path, index=False)
    print(f"Results have been saved to '{csv_path}'")

if __name__ == "__main__":
    csv_output_path = 'jax_benchmark_results.csv'
    jax_dot(csv_output_path)