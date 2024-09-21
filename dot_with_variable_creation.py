import torch.func as func
import torch
from utils import benchmark

try: 
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed, not benchmarking JAX")



NUMBER_OF_TESTS = 1000
n = 10000
m = 100

# ====== comparing the dot product performance =======  
def dot_product_simple_torch():
    a = torch.randn(m, n)
    b = torch.randn(m, n)
    result_dot = torch.empty(m, 1)
    for i in range(m):
        result_dot[i] = torch.dot(a[i], b[i])
    return result_dot

def dot_product_torch_vmap():
    a = torch.randn(m, n)
    b = torch.randn(m, n)
    vec_dot = func.vmap(torch.dot)
    return vec_dot(a, b)

def dot_product_simple_jax():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, shape=(m, n))
    b = jax.random.normal(key, shape=(m, n))
    result_dot = jnp.empty((m, 1))
    for i in range(m):
        result_dot.at[i].set(jnp.dot(a[i], b[i]))
    return result_dot

def dot_product_jax_vmap():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, shape=(m, n))
    b = jax.random.normal(key, shape=(m, n))
    vec_dot = jax.vmap(jnp.dot)
    return vec_dot(a, b)




# Apply benchmark decorator to functions
@benchmark(NUMBER_OF_TESTS)
def benchmark_torch():
    return dot_product_simple_torch()

@benchmark(NUMBER_OF_TESTS)
def benchmark_torch_vmap():
    return dot_product_torch_vmap()

@benchmark(NUMBER_OF_TESTS)
def benchmark_jax():
    return dot_product_simple_jax()

@benchmark(NUMBER_OF_TESTS)
def benchmark_jax_vmap():
    return dot_product_jax_vmap()



if __name__ == "__main__":
    # set jax to use cpu only
    jax.config.update('jax_platform_name', 'cpu')
    jax_result, jax_mean, jax_std, jax_memory, jax_memory_std = benchmark_jax()
    jax_vmap_result, jax_vmap_mean, jax_vmap_std, jax_vmap_memory, jax_vmap_memory_std = benchmark_jax_vmap()
    # Run benchmarks and capture results
    torch_result, torch_mean, torch_std, torch_memory, torch_memory_std = benchmark_torch()
    torch_vmap_result, torch_vmap_mean, torch_vmap_std, torch_vmap_memory, torch_vmap_memory_std = benchmark_torch_vmap()

    # Store results for plotting
    benchmark_results = {
        'Torch': (torch_mean, torch_std, torch_memory),
        'Torch_VMAP': (torch_vmap_mean, torch_vmap_std, torch_vmap_memory),
        'JAX': (jax_mean, jax_std, jax_memory),
        'JAX_VMAP': (jax_vmap_mean, jax_vmap_std, jax_vmap_memory)
    }

    import plotly.express as px
    import pandas as pd

    # Create a DataFrame from the benchmark results
    bench_pd = pd.DataFrame({
        'Framework': ['Torch', 'Torch_VMAP', 'JAX', 'JAX_VMAP'],
        'Mean Time': [torch_mean, torch_vmap_mean, jax_mean, jax_vmap_mean],
        'Std Time': [torch_std, torch_vmap_std, jax_std, jax_vmap_std],
        'Mean Memory': [torch_memory, torch_vmap_memory, jax_memory, jax_vmap_memory],
        'Std Memory': [torch_memory_std, torch_vmap_memory_std, jax_memory_std, jax_vmap_memory_std]
    })

    # Create and show the time comparison plot
    fig_time = px.bar(bench_pd, x='Framework', y='Mean Time', error_y='Std Time',
                      title='Time Comparison')
    fig_time.show()

    # Create and show the memory comparison plot
    fig_memory = px.bar(bench_pd, x='Framework', y='Mean Memory',
                        title='Memory Comparison')
    fig_memory.show()


    # second plot comparing the mean memory
