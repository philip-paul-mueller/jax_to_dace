import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append("..")
from JaxprToSDFG import JaxprToSDFG

# This must be enabled when `make_jaxpr` is called, because otherwhise we get problems.
jax.config.update("jax_enable_x64", True)

N1, N2, N3 = 20, 20, 4
A = np.random.rand(N1, N2).astype(np.float64)
B = np.random.rand(N1, N2).astype(np.float64)
C = np.random.rand(N1).astype(np.float64)
D = np.random.rand(N1).astype(np.float64)

def f1(array_like_0, array_like_1):
    return jnp.dot(array_like_0, array_like_1)

t = JaxprToSDFG()

ARGS_ = ((A, B), (C, D), (A, C), (C, A))
for args_ in ARGS_:
    print(f"Shapes -> array_like_0: {args_[0].shape}, array_like_1: {args_[1].shape}")
    f1_jaxpr = jax.make_jaxpr(f1)(*args_)
    f1_sdfg = t(f1_jaxpr)
    #f1_sdfg.view()
    _OUT = f1_sdfg(*args_)

    resExp = f1(*args_)
    resDC  = _OUT

    assert np.all(np.abs(resDC - resExp) <= 10**(-13))
