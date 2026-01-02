# Environment Setup Instructions

## Option 1: Using pip (Recommended for most users)

1. **Create a new virtual environment:**
   ```bash
   python -m venv bi_rnn_env
   source bi_rnn_env/bin/activate  # On Windows: bi_rnn_env\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the local package:**
   ```bash
   pip install -e CogModelingRNNsTutorial/
   ```

## Option 2: Using conda

1. **Create a new conda environment:**
   ```bash
   conda create -n bi_rnn_env python=3.9
   conda activate bi_rnn_env
   ```

2. **Install JAX with CUDA support (if you have a GPU):**
   ```bash
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

   Or CPU-only:
   ```bash
   pip install --upgrade jax jaxlib
   ```

3. **Install other requirements:**
   ```bash
   pip install dm-haiku optax matplotlib pandas scipy seaborn plotnine requests gdown notebook ipykernel tqdm
   ```

4. **Install the local package:**
   ```bash
   pip install -e CogModelingRNNsTutorial/
   ```

## Option 3: Quick install script

Run this script in your terminal:
```bash
#!/bin/bash
# Create and activate virtual environment
python -m venv bi_rnn_env
source bi_rnn_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX (CPU version)
pip install jax jaxlib

# Install all other requirements
pip install -r requirements.txt

# Install local package
pip install -e CogModelingRNNsTutorial/

echo "Setup complete! Activate the environment with: source bi_rnn_env/bin/activate"
```

## Verify Installation

After installation, run this to verify everything works:
```python
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print("All packages imported successfully!")
```

## Common Issues

1. **JAX installation issues**: JAX has different versions for CPU/GPU. Make sure to install the correct version for your system.

2. **Apple Silicon (M1/M2)**: Use these commands:
   ```bash
   pip install --upgrade pip
   pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

3. **Permission errors**: Make sure your virtual environment is in a directory you have write permissions for.

## Running the Notebook

After setup:
1. Activate your environment
2. Start Jupyter:
   ```bash
   jupyter notebook test_bi_control_rnn.ipynb
   ```
