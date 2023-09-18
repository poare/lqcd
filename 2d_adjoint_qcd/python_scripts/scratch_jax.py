import numpy as jnp
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import cond
import time

N = 1000
a = np.array([0, 1])
b = np.array([0, 2])
aj = jnp.array([0, 1])
bj = jnp.array([0, 2])

def time_numpy(N):
    for i in range(N):
        if np.array_equal(a, b):
            x = 2
        else:
            x = 1
start = time.time()
time_numpy(N)
print(f'Numpy time for {N} repetitions is: {time.time() - start}')

def time_jax_numpy(N):
    # for i in jnp.arange(N):
    #     cond(jnp.array_equal(aj, bj), lambda : 2, lambda : 1)
    jax.lax.while_loop(lambda i : i < N,
        print(i),
        # lambda i : cond(jnp.array_equal(aj, bj), lambda : 2, lambda : 1),
        0
    )
start = time.time()
time_jax_numpy(N)
print(f'Jax numpy time for {N} repetitions is: {time.time() - start}')

time_jit_np = jax.jit(time_jax_numpy)
start = time.time()
time_jit_np(N)
print(f'Jax numpy time for {N} repetitions with JIT is: {time.time() - start}')