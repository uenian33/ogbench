# request a short GPU slot and run Python inline
srun --gpus=1 --time=00:05:00 --mem=600M --cpus-per-task=2 bash -lc '
module load mamba
source activate ogbench
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<PY
import jax, jax.numpy as jnp
print("jax:", jax.__version__)
from jax.lib import xla_bridge
print("backend:", xla_bridge.get_backend().platform)
print("devices:", jax.devices())
# do a tiny GPU compute and show which device it ran on
x = jnp.ones((2048, 2048))
y = x @ x
y.block_until_ready()
print("result device:", y.device())
print("GPU present?:", any(d.platform in ("gpu","cuda") for d in jax.devices()))
PY
'
