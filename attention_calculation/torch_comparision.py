import torch
import torch.nn as torch_nn
import math
import pandas as pd
from utils import benchmark

NUMBER_OF_TESTS = 100

@benchmark(NUMBER_OF_TESTS)
def scaled_dot_product_attention_torch(qkv):
    d_k = qkv.shape[-1] // 3
    q, k, v = torch.split(qkv, [d_k, d_k, d_k], dim=-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = torch_nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def torch_attention_benchmark(csv_path: str, device: str = 'cpu'):
    # Set PyTorch to use the specified device
    torch_device = torch.device(device)
    
    qkv_torch = torch.randn(10, 10, 512 * 3, device=torch_device)

    _, mean_time_torch, std_dev_time_torch, mean_memory_torch, std_dev_memory_torch, mean_GPU_memory_torch = scaled_dot_product_attention_torch(qkv_torch)

    results_df = pd.DataFrame({
        'Framework': ['PyTorch'],
        'Mean Time': [mean_time_torch],
        'Std Time': [std_dev_time_torch],
        'Mean Memory': [mean_memory_torch],
        'Std Memory': [std_dev_memory_torch],
        'Mean GPU Memory': [mean_GPU_memory_torch],
    })

    results_df.to_csv(csv_path, index=False)
    print(f"PyTorch results have been saved to '{csv_path}'")

if __name__ == "__main__":
    csv_output_path = 'torch_attention_benchmark_results.csv'
    torch_attention_benchmark(csv_output_path)