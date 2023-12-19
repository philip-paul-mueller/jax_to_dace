# Info
This code is able to translate [Jax](https://github.com/google/jax) [code](https://jax.readthedocs.io/en/latest/jaxpr.html) into [SDFG](https://github.com/spcl/dace).
For more information on how to use it, see [First_translation.ipynb](./First_translation.ipynb).      
If you make your own tests, please create a new file for this, merging will be simpler.

## Dependencies
- `JAX`: Vanilla version.
- `gt4py`: A Version that contains commit `ed77c4` (new argument order of SDFG).
- `DaCe`: A version that contains commit `206be9` (or the `_fast_call()` routine of the `CompiledSDFG`).


# Design
The code is designed modular, the class `JaxprToSDFG` acts as some kind of driver.
As the name implies it does not translate the individual Jax Equations to SDFG, instead it delegates this to an appropriate translator.

A translator is a class that is derived from `JaxIntrinsicTranslatorInterface` and should be located in `JaxprToSDFG/translators`.
A translator should handle as many Jax primitives as possible as long as they are strongly related to promote code reuse.
However, it is perfectly fine if a translator only handles one intrinsic.

To register a new translator add it to `JaxprToSDFG._initEqnTranslators()`.


## Todos
There are some todos, use `grep -i todo` to get an overview.


### Lists of Intrinsics
Here is a list of intrinsics, taken from `https://github.com/PierrickPochelu/JaxDecompiler/blob/main/src/JaxDecompiler/primitive_mapping.py`:

Not yet implemented, sort roughly according to importances:
- [ ] `scatter_add`
- [ ] `while`
- [ ] `fori`
- [ ] `cond` (conditional)
- [ ] `scan`
- [ ] `clamp`
- [ ] `copy`
- [ ] `random_seed`
- [ ] `random_unwrap`
- [ ] `random_wrap`
- [ ] `random_bits`
- [ ] `shift_right_logical`
- [ ] `shift_left_logical`
- [ ] `sort`
- [ ] `xla_pmap`
- [ ] `xla_call`
- [ ] `rev`
- [ ] `conv_general_dilated`
- [ ] `dynamic_slice`
- [ ] `dynamic_update_slice`
- [ ] `bitcast_convert_type`
- [ ] `erf_inv`
- [ ] `stop_gradient`
- [ ] `transpose`
- [ ] `iota`
- [ ] `coo_fromdense`
- [ ] `coo_matvec`


Already implemented intrinsics (most of them are arithmetic operations, which are handled by `SimpleTranslator`)
- [x] `add`
- [x] `add_any`
- [x] `mul`
- [x] `sub`
- [x] `neg`
- [x] `div`
- [x] `rem`
- [x] `floor`
- [x] `ceil`
- [x] `round`
- [x] `integer_pow`
- [x] `pow`
- [x] `sqrt`
- [x] `log`
- [x] `exp`
- [x] `dot_general`
- [x] `cos`
- [x] `sin`
- [x] `tan`
- [x] `tanh`
- [x] `acos`
- [x] `asin`
- [x] `atan`
- [x] `convert_element_type`
- [x] `reshape`
- [x] `gather`
- [x] `concatenate`
- [x] `squeeze`
- [x] `argmin`
- [x] `argmax`
- [x] `min`
- [x] `reduce_min`
- [x] `max`
- [x] `reduce_max`
- [x] `abs`
- [x] `sign`
- [x] `reduce_sum`
- [x] `broadcast_in_dim`
- [x] `select_n`
- [x] `ne`
- [x] `eq`
- [x] `ge`
- [x] `gt`
- [x] `le`
- [x] `lt`
- [x] `reduce_or`
- [x] `reduce_and`
- [x] `slice`
- [x] `or__`
- [x] `and__`
- [x] `not__`


