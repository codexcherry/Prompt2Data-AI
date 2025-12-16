#!/usr/bin/env python3
"""Test matplotlib installation and backend"""

import sys
print(f"Python version: {sys.version}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    matplotlib.use('Agg')  # Non-interactive backend
    print("✓ Matplotlib backend set to 'Agg'")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Simple test plot
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('Test Plot')
    
    output_file = 'report/images/test_plot.png'
    import os
    os.makedirs('report/images', exist_ok=True)
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    if os.path.exists(output_file):
        print(f"✓ Test plot saved successfully: {output_file}")
        print("✓ Matplotlib is working correctly!")
    else:
        print(f"✗ Test plot file not found: {output_file}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

