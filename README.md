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


### Lists of Intrinsics
Here is a list of intrinsics, taken from `https://github.com/PierrickPochelu/JaxDecompiler/blob/main/src/JaxDecompiler/primitive_mapping.py`:

- [x] `add`
- [ ] `add_any`
- [x] `mul`
- [x] `sub`
- [x] `neg`
- [x] `div`
- [x] `rem`
- [x] `floor`
- [x] `ceil`
- [x] `round`
- [ ] `clamp`
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
- [ ] `copy`
- [x] `convert_element_type`
- [ ] `reshape`
- [x] `gather`
- [ ] `random_seed`
- [ ] `random_unwrap`
- [ ] `random_wrap`
- [ ] `random_bits`
- [ ] `shift_right_logical`
- [ ] `shift_left_logical`
- [x] `concatenate`
- [x] `squeeze`
- [ ] `argmin`
- [ ] `argmax`
- [ ] `min`
- [ ] `reduce_min`
- [ ] `max`
- [ ] `reduce_max`
- [x] `abs`
- [x] `sign`
- [ ] `reduce_sum`
- [x] `broadcast_in_dim`
- [x] `select_n`
- [x] `ne`
- [x] `eq`
- [x] `ge`
- [x] `gt`
- [x] `le`
- [x] `lt`
- [ ] `sort`
- [ ] `reduce_or`
- [ ] `reduce_and`
- [ ] `xla_pmap`
- [ ] `xla_call`
- [ ] `rev`
- [ ] `conv_general_dilated`
- [ ] `dynamic_slice`
- [x] `slice`
- [ ] `dynamic_update_slice`
- [ ] `scatter_add`
- [ ] `or__`
- [ ] `and__`
- [ ] `bitcast_convert_type`
- [ ] `erf_inv`
- [ ] `stop_gradient`
- [ ] `transpose`
- [ ] `iota`
- [ ] `coo_fromdense`
- [ ] `coo_matvec`
- [ ] `scan`

