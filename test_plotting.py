import matplotlib
matplotlib.use('Qt5Agg')  # Try Qt backend instead
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Backend: {matplotlib.get_backend()}")

import numpy as np
print(f"NumPy version: {np.__version__}")

import matplotlib.pyplot as plt

try:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
    ax.set_title('Simple Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()