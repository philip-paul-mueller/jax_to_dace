# Info
This code is able to translate [Jax](https://github.com/google/jax) [code](https://jax.readthedocs.io/en/latest/jaxpr.html) into [SDFG](https://github.com/spcl/dace).
For more information on how to use it, see [First_translation.ipynb](./First_translation.ipynb).      
If you make your own tests, please create a new file for this, merging will be simpler.

# Design
The code is designed modular, the class `JaxprToSDFG` acts as some kind of driver.
As the name implies it does not translate the individual Jax Equations to SDFG, instead it delegates this to an appropriate translator.

A translator is a class that is derived from `JaxIntrinsicTranslatorInterface` and should be located in `JaxprToSDFG/translators`.
A translator should handle as many Jax primitives as possible as long as they are strongly related to promote code reuse.
However, it is perfectly fine if a translator only handles one intrinsic.

To register a new translator add it to `JaxprToSDFG._initEqnTranslators()`.


## Todos
There are some todos, use `grep -i todo` to get an overview.





