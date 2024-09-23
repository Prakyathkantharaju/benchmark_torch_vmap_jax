import os
from dot_product.torch_comparision import torch_dot
from dot_product.jax_comparision import jax_dot
from attention_calculation.torch_comparision import torch_attention_benchmark
from attention_calculation.jax_comparision import jax_attention_benchmark
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    # Create the comparison_data folder if it doesn't exist
    os.makedirs('comparison_data', exist_ok=True)

    # Run PyTorch comparison
    torch_csv_path = os.path.join('comparison_data', 'torch_dot_benchmark_results.csv')
    torch_dot(torch_csv_path, device='cpu')

    # Run JAX comparison
    jax_csv_path = os.path.join('comparison_data', 'jax_dot_benchmark_results.csv')
    jax_dot(jax_csv_path, device='cpu', warmup=True)

    # Run PyTorch attention benchmark
    torch_attention_csv_path = os.path.join('comparison_data', 'torch_attention_benchmark_results.csv')
    torch_attention_benchmark(torch_attention_csv_path, device='cpu')

    # Run JAX attention benchmark
    jax_attention_csv_path = os.path.join('comparison_data', 'jax_attention_benchmark_results.csv')
    jax_attention_benchmark(jax_attention_csv_path, device='cpu', warmup=True)


    # Load the data from the CSV files
    torch_dot_data = pd.read_csv(torch_csv_path)
    jax_dot_data = pd.read_csv(jax_csv_path)
    torch_attention_data = pd.read_csv(torch_attention_csv_path)
    jax_attention_data = pd.read_csv(jax_attention_csv_path)


    # Set the style and color palette
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Dot product comparison
    dot_data = pd.concat([torch_dot_data, jax_dot_data])
    colors_torch = ['#FF6666'] * len(torch_dot_data)
    colors_jax = ['#6666FF'] * len(jax_dot_data)
    colors_1 = colors_torch + colors_jax
    sns.barplot(x='Framework', y='Mean Time', data=dot_data, ax=ax1, palette=colors_1)
    ax1.set_title('Dot Product Comparison', fontsize=16)
    ax1.set_xlabel('Framework', fontsize=12)
    ax1.set_ylabel('Mean Time (s)', fontsize=12)

    # Attention calculation comparison
    attention_data = pd.concat([torch_attention_data, jax_attention_data])
    colors_torch = ['#FF6666'] * len(torch_attention_data)
    colors_jax = ['#6666FF'] * len(jax_attention_data)
    colors_2 = colors_torch + colors_jax
    sns.barplot(x='Framework', y='Mean Time', data=attention_data, ax=ax2, palette=colors_2)
    ax2.set_title('Attention Calculation Comparison', fontsize=16)
    ax2.set_xlabel('Framework', fontsize=12)
    ax2.set_ylabel('Mean Time (s)', fontsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('comparison_data/framework_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Memory comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Dot product comparison
    sns.barplot(x='Framework', y='Mean Memory', data=dot_data, ax=ax1, palette=colors_1)
    ax1.set_title('Dot Product Comparison', fontsize=16)
    ax1.set_xlabel('Framework', fontsize=12)
    ax1.set_ylabel('Mean Memory (KB)', fontsize=12)

    # Attention calculation comparison
    sns.barplot(x='Framework', y='Mean Memory', data=attention_data, ax=ax2, palette=colors_2)
    ax2.set_title('Attention Calculation Comparison', fontsize=16)
    ax2.set_xlabel('Framework', fontsize=12)
    ax2.set_ylabel('Mean Memory (KB)', fontsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('comparison_data/framework_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    main()
