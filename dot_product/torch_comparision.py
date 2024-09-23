import torch
import torch.func as func
from utils import benchmark
import pandas as pd

NUMBER_OF_TESTS = 100
n, m = 10000, 100

@benchmark(NUMBER_OF_TESTS)
def dot_product_simple_torch(a: torch.Tensor, b: torch.Tensor):
    result_dot = torch.empty(m, 1)
    for i in range(m):
        result_dot[i] = torch.dot(a[i], b[i])
    return result_dot

@benchmark(NUMBER_OF_TESTS)
def dot_product_torch_vmap(a: torch.Tensor, b: torch.Tensor):
    return func.vmap(torch.dot)(a, b)


def torch_dot(csv_path: str, device : str = 'cpu'):
    a = torch.randn(m, n, device=device)
    b = torch.randn(m, n, device=device)
    
    torch_result, torch_mean, torch_std, torch_memory, torch_memory_std, torch_GPU_memory = dot_product_simple_torch(a, b)
    torch_vmap_result, torch_vmap_mean, torch_vmap_std, torch_vmap_memory, torch_vmap_memory_std, torch_vmap_GPU_memory = dot_product_torch_vmap(a, b)
    # Create a DataFrame from the benchmark results
    results_df = pd.DataFrame({
        'Framework': ['Torch', 'Torch VMAP'],
        'Mean Time': [torch_mean, torch_vmap_mean],
        'Std Time': [torch_std, torch_vmap_std],
        'Mean Memory': [torch_memory, torch_vmap_memory],
        'Std Memory': [torch_memory_std, torch_vmap_memory_std],
        'Mean GPU Memory': [torch_GPU_memory, torch_vmap_GPU_memory],
    })

    # Save results to CSV
    results_df.to_csv(csv_path, index=False)
    print(f"Results have been saved to '{csv_path}'")

if __name__ == "__main__":
    csv_output_path = 'torch_benchmark_results.csv'
    torch_dot(csv_output_path)