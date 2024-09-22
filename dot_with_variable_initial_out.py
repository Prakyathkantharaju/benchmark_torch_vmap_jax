import torch.func as func
import torch
from utils import benchmark
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

try: 
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed, not benchmarking JAX")



# NOTE: the profiler will only produce result it will also results in profiler information.
NUMBER_OF_TESTS = 1000
n = 10000
m = 100

# ====== comparing the dot product performance =======  
def dot_product_simple_torch(a: torch.Tensor, b: torch.Tensor):
    result_dot = torch.empty(m, 1)
    for i in range(m):
        result_dot[i] = torch.dot(a[i], b[i])
    return result_dot

def dot_product_torch_vmap(a: torch.Tensor, b: torch.Tensor):
    vec_dot = func.vmap(torch.dot)
    return vec_dot(a, b)

def dot_product_simple_jax(a: jax.Array, b: jax.Array):
    result_dot = jnp.empty((m, 1))
    for i in range(m):
        result_dot.at[i].set(jnp.dot(a[i], b[i]))
    return result_dot

def dot_product_jax_vmap(a: jax.Array, b: jax.Array):
    vec_dot = jax.vmap(jnp.dot)
    return vec_dot(a, b)




# Apply benchmark decorator to functions
@benchmark(NUMBER_OF_TESTS)
def benchmark_torch(a: torch.Tensor, b: torch.Tensor):
    return dot_product_simple_torch(a, b)

@benchmark(NUMBER_OF_TESTS)
def benchmark_torch_vmap(a: torch.Tensor, b: torch.Tensor):
    return dot_product_torch_vmap(a, b)

@benchmark(NUMBER_OF_TESTS)
def benchmark_jax(a: jax.Array, b: jax.Array):
    return dot_product_simple_jax(a, b)

@benchmark(NUMBER_OF_TESTS)
def benchmark_jax_vmap(a: jax.Array, b: jax.Array):
    return dot_product_jax_vmap(a, b)



if __name__ == "__main__":
    # set jax to use cpu only
    jax.config.update('jax_platform_name', 'cpu')
    random_key = jax.random.PRNGKey(0)
    a = jax.random.normal(random_key, shape=(m, n))
    b = jax.random.normal(random_key, shape=(m, n))
    jax_result, jax_mean, jax_std, jax_memory, jax_memory_std = benchmark_jax(a, b)
    jax_vmap_result, jax_vmap_mean, jax_vmap_std, jax_vmap_memory, jax_vmap_memory_std = benchmark_jax_vmap(a, b)
    # Run benchmarks and capture results
    a = torch.randn(m, n)
    b = torch.randn(m, n)
    torch_result, torch_mean, torch_std, torch_memory, torch_memory_std = benchmark_torch(a, b)
    torch_vmap_result, torch_vmap_mean, torch_vmap_std, torch_vmap_memory, torch_vmap_memory_std = benchmark_torch_vmap(a, b)


    # Create a DataFrame from the benchmark results
    bench_pd = pd.DataFrame({
        'Framework': ['Torch', 'Torch VMAP', 'JAX', 'JAX VMAP'],
        'Mean Time': [torch_mean, torch_vmap_mean, jax_mean, jax_vmap_mean],
        'Std Time': [torch_std, torch_vmap_std, jax_std, jax_vmap_std],
        'Mean Memory': [torch_memory, torch_vmap_memory, jax_memory, jax_vmap_memory],
        'Std Memory': [torch_memory_std, torch_vmap_memory_std, jax_memory_std, jax_vmap_memory_std]
    })

    # Set the style and color palette
    sns.set_style("whitegrid")
    colors = ['#FF0000', '#FF6666', '#6666FF', '#9933FF']

    # Create time comparison plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Framework', y='Mean Time', hue='Framework', data=bench_pd, palette=colors, legend=False)
    ax.errorbar(x=ax.get_xticks(), y=bench_pd['Mean Time'], yerr=bench_pd['Std Time'], fmt='none', c='black', capsize=5)

    plt.title('Time Comparison', fontsize=16)
    plt.xlabel('Framework', fontsize=12)
    plt.ylabel('Mean Time (s)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create memory comparison plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Framework', y='Mean Memory', hue='Framework', data=bench_pd, palette=colors, legend=False)

    plt.title('Memory Comparison', fontsize=16)
    plt.xlabel('Framework', fontsize=12)
    plt.ylabel('Mean Memory Usage (MB)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("Plots have been displayed. Close the plot windows to end the script.")
