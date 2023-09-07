################################################################################
# Playground for testing Jax's autodiff and how it works with complex input.   # 
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

import numpy as np
import itertools
import scipy
from scipy.sparse import bsr_matrix, csr_matrix

# for complex autodiff
import jax.numpy as jnp
from jax import grad, jit, vmap

from jax import random
key = random.PRNGKey(0)

import rhmc

def is_equal_jax(A, B, eps = rhmc.EPS):
    """Returns whether two jax arrays are equal up to precision eps."""
    return jnp.allclose(A, B, eps)

A_key = random.split(key, 1)
A = random.normal(A_key, (3, 3))

# Example: derivative of trace with respect to A. Note that tr(A) = A00 + A11 + A22, so 
# grad(tr)(A) = id.
grad_tr = grad(jnp.trace)
print(is_equal_jax(grad_tr(A), jnp.eye(3)))

# gradient of Tr[A @ A] = Tr[A00^2 + A01*A10 + ...] is 2 A^transpose.
grad_square = grad(lambda W : jnp.trace(W @ W))
print(is_equal_jax(grad_square(A), 2*A.transpose()))


