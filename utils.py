import time
from functools import wraps
import numpy as np
import psutil
import os
import gc

NUMBER_OF_TESTS = 1000


GPU_TESTING = False
if os.environ.get('CUDA_TESTING'):

    # if CUDA_DEVICE is set, then we are running on GPU
    import pynvml
    pynvml.nvmlInit()
    GPU_DEVICE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_TESTING = True


# Benchmark decorator
def benchmark(runs=NUMBER_OF_TESTS):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            memory_usages = []
            GPU_memory_usages = []
            
            process = psutil.Process(os.getpid())
            
            for _ in range(runs):
                # Memory usage before
                mem_before = process.memory_info().rss
                if GPU_TESTING:
                    GPU_memory_before = pynvml.nvmlDeviceGetMemoryInfo(GPU_DEVICE).used
                
                # Time measurement
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                # Memory usage after
                mem_after = process.memory_info().rss
                if GPU_TESTING:
                    GPU_memory_after = pynvml.nvmlDeviceGetMemoryInfo(GPU_DEVICE).used
                    GPU_memory_used = GPU_memory_after - GPU_memory_before
                    GPU_memory_usages.append(GPU_memory_used)
                else:
                    GPU_memory_usages.append(0)
                
                mem_used = mem_after - mem_before
                
                times.append(end - start)
                memory_usages.append(mem_used)
                gc.collect()
            
            mean_time = np.mean(times)
            std_dev_time = np.std(times)
            mean_memory = np.mean(memory_usages)
            std_dev_memory = np.std(memory_usages)
            mean_GPU_memory = np.mean(GPU_memory_usages)
            print(f"{func.__name__}:")
            print(f"  Mean time: {mean_time:.6f} seconds")
            print(f"  Time std dev: {std_dev_time:.6f} seconds")
            print(f"  Mean memory usage: {mean_memory / 1024:.2f} KB")
            print(f"  Memory std dev: {std_dev_memory / 1024:.2f} KB")
            
            return result, mean_time, std_dev_time, mean_memory / 1024, std_dev_memory / 1024, mean_GPU_memory / 1024
        return wrapper
    return decorator