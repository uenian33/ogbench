# request a short GPU slot and run Python inline
srun --gpus=1 --time=00:05:00 --mem=600M --cpus-per-task=2 bash -lc '
module load mamba
source activate ogbench
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<PY
import jax, jax.numpy as jnp
print("jax:", jax.__version__)
print("devices:", jax.devices())           # should list CudaDevice(...)
x = jnp.ones((2048, 2048))
y = (x @ x).block_until_ready()
print("ran on:", y.device)                 # <-- no parentheses

PY
'
