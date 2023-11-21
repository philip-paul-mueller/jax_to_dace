import sys
import numpy as np
import jax
import jax.numpy as jnp
import dace

sys.path.append("..")
from JaxprToSDFG import JaxprToSDFG

# This must be enabled when `make_jaxpr` is called, because otherwhise we get problems.
jax.config.update("jax_enable_x64", True)


N1, N2 = 20, 20
A = np.random.rand(N1, N2).astype(np.float64)


def f_reshape(array_like_0, newshape=(N1*N2,)):
    return jnp.reshape(array_like_0, newshape)

def f_min(array_like_0, axis=None):
    return jnp.min(array_like_0, axis=axis)

def f_max(array_like_0, axis=None):
    return jnp.max(array_like_0, axis=axis)

def f_sum(array_like_0, axis=None):
    return jnp.sum(array_like_0, axis=axis)

def f_prod(array_like_0, axis=None):
    return jnp.prod(array_like_0, axis=axis)

def f_argmin(array_like_0, axis=0):
    return jnp.argmin(array_like_0, axis=axis)

def f_argmax(array_like_0, axis=0):
    return jnp.argmax(array_like_0, axis=axis)


funcs = (f_reshape, f_min, f_max, f_sum, f_prod, f_argmin, f_argmax)


t = JaxprToSDFG()
for func in funcs:
    print(f"Testing function: {func}")
    
    func_jaxpr = jax.make_jaxpr(func)(A)
    print(func_jaxpr)
    f1_sdfg = t(func_jaxpr)

    resExp = func(A)
    resDC  = f1_sdfg(A)
    assert np.all(np.abs(resDC - resExp) <= 10**(-9)), f"{resExp} vs {resDC}"
