"""This file contains everything that is related to reduction operations.
"""
from JaxprToSDFG import JaxprToSDFG
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

import ast
from functools import reduce
import copy
from numbers import Integral
from typing import Any, Callable, List, Tuple, Union

from jax._src.core import JaxprEqn

import dace
from dace import dtypes, subsets
from dace.memlet import Memlet
from dace.frontend.python.common import StringLiteral
from dace.frontend.python import astutils
from dace.symbolic import pystr_to_symbolic
from dace.frontend.python.nested_call import NestedCall


class ReductionTranslator(JaxIntrinsicTranslatorInterface):
    """This class handles reduction operations

    It forwards the call to dace.
    """
    __slots__ = ("m_primitives", )


    def __init__(self):
        """Initialization
        """
        super().__init__()      # As requiered call the initializer of the super class
        self.m_primitives = ["reduce_min", "reduce_max", "reduce_sum", "reduce_prod", "argmin", "argmax"]
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        
        return eqn.primitive.name in self.m_primitives
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        Notes:
            -
        """
        if(len(eqn.invars) != 1):
            raise ValueError(f"Reduction operations needs one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        if eqn.primitive.name in ["argmin", "argmax"] and len(translator.getArray(inVarNames[0]).shape) == 1:
            raise ValueError(f"Dace seems to have an issue with flattened arrays for _argminmax")
        #

        # Some information about the variables we are working with.
        inAVal   = eqn.invars[0].aval
        inShape  = inAVal.shape
        outAVal  = eqn.outvars[0].aval
        outShape = outAVal.shape

        # Along these axis we have to reduce.
        red_axes = eqn.params["axes"]

        redfunction = None
        redOps = None
        identity = None
        if eqn.primitive.name == "reduce_min":
            redfunction = "lambda x, y: min(x, y)"
            identity = dtypes.max_value(translator.getArray(inVarNames[0]).dtype)
            redOps = 'min({}, {})'
        elif eqn.primitive.name == "reduce_max":
            redfunction = "lambda x, y: max(x, y)"
            identity = dtypes.min_value(translator.getArray(inVarNames[0]).dtype)
            redOps = 'max({}, {})'
        elif eqn.primitive.name == "reduce_sum":
            redfunction = "lambda x, y: x + y"
            identity = 0.
            redOps = '(({}) + ({}))'
        elif eqn.primitive.name == "reduce_prod":
            redfunction = "lambda x, y: x * y"
            identity = 1.
            redOps = '(({}) * ({}))'
        elif eqn.primitive.name == "argmin":
            redfunction = "min"
        elif eqn.primitive.name == "argmax":
            redfunction = "max"
        
        if not redfunction:
            raise NotImplementedError(f"This reduction primitive [{eqn.primitive.name}] is not implemented")

        if((redOps is not None) and (len(red_axes) == 1) and (len(inShape) > 1)):
            # Specialization we perform reduction along one axis and there are more than one axis remains.
            #  This is basically that we "sum" along one axis.
            assert len(outShape) > 0
            assert len(red_axes) == 1
            assert inShape[red_axes[0]] > 1
            red_axe = red_axes[0]    # Since we have only one element.

            tMapRanges, tOutputs_, tInputs_ = [], [], []
            for dim, size in enumerate(inShape):
                if(dim == red_axe):
                    tInputs_.append( None )
                else:
                    tMapRanges.append( (f'__i{dim}', f'0:{size}') )
                    tOutputs_.append( tMapRanges[-1][0] )
                    tInputs_.append( tMapRanges[-1][0] )
            #

            # Now we build the output memlet.
            tOutputs = dict(__out=dace.Memlet.simple(outVarNames[0], ', '.join(tOutputs_)))

            # Now we build the input memlets, we need one for every element in that dimension.
            tInputs = dict()
            tInpArgs = []
            assert tInputs_[red_axe] is None
            assert sum([1 if x is None else 0  for x in tInputs_]) == 1
            for i in range(inShape[red_axe]):
                tInpArgs.append(f'__in{i}')
                tInputs_[red_axe] = str(i)
                tInputs[tInpArgs[-1]] = dace.Memlet.simple(inVarNames[0], ', '.join(tInputs_))
            #
            tInputs_[red_axe] = None

            # Now we must write the tasklet which we will also use correct paranthesis.
            while len(tInpArgs) != 1:
                tRedInpArgs = []
                while len(tInpArgs) >= 2:
                    arg1 = tInpArgs.pop()
                    arg2 = tInpArgs.pop()
                    redArg = redOps.format(arg1, arg2)
                    tRedInpArgs.append(redArg)
                if(len(tInpArgs) == 1):
                    tRedInpArgs.append(tInpArgs.pop())
                assert len(tInpArgs) == 0
                tInpArgs = tRedInpArgs ; del tRedInpArgs
            #
            tCode = f'__out = {tInpArgs[0]}'

            eqnState.add_mapped_tasklet(
                f'_reduction_{outVarNames[0]}',
                map_ranges=tMapRanges,
                inputs=tInputs,
                code=tCode,
                outputs=tOutputs,
                external_edges=True
            )

        elif identity is not None:
            _reduce(None,
                    translator.getSDFG(),
                    eqnState,
                    redfunction,
                    inVarNames[0],
                    outVarNames[0],
                    axis=eqn.params["axes"],
                    identity=identity
                    )
        else:
            nest, out = _argminmax(None,
                       translator.getSDFG(),
                       eqnState,
                       inVarNames[0],
                       eqn.params["axes"][0],
                       redfunction,
                       result_type=JaxprToSDFG._translateDType(eqn.params["index_dtype"].name)
            )
            # _argminmax generates multiple states.
            # Just keep the last one as tracked in nest obj.
            if nest:
                eqnState = nest.last_state
            
            # save the result to the output var
            output_subset = subsets.Range.from_array(translator.getArray(outVarNames[0]))
            output_memlet = dace.Memlet.simple(outVarNames[0], output_subset)
            inpnode = eqnState.add_read(out)
            outnode = eqnState.add_write(outVarNames[0])
            eqnState.add_nedge(inpnode, outnode, output_memlet)

        return eqnState
    # end def: translateEqn


# end class(DotGeneralTranslator):


# Functions below:
# copied from Dace (commit 12b998193d966ce656384aaba9dfd32395a4d42c)

def normalize_axes(axes: Tuple[int], max_dim: int) -> List[int]:
    """ Normalize a list of axes by converting negative dimensions to positive.

        :param dims: the list of dimensions, possibly containing negative ints.
        :param max_dim: the total amount of dimensions.
        :return: a list of dimensions containing only positive ints.
    """

    return [ax if ax >= 0 else max_dim + ax for ax in axes]

def _reduce(pv: Any,
            sdfg: dace.SDFG,
            state: dace.SDFGState,
            redfunction: Callable[[Any, Any], Any],
            in_array: str,
            out_array=None,
            axis=None,
            identity=None):
    if out_array is None:
        inarr = in_array
        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_shape = None

        # check if we are reducing along all axes
        if axis is not None and len(axis) == len(input_subset.size()):
            reduce_all = all(x == y for x, y in zip(axis, range(len(input_subset.size()))))
        else:
            reduce_all = False

        if axis is None or reduce_all:
            output_shape = [1]
        else:
            output_subset = copy.deepcopy(input_subset)
            output_subset.pop(axis)
            output_shape = output_subset.size()
        if (len(output_shape) == 1 and output_shape[0] == 1):
            outarr = sdfg.temp_data_name()
            outarr, arr = sdfg.add_scalar(outarr, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage, transient=True)
        else:
            outarr, arr = sdfg.add_temp_transient(output_shape, sdfg.arrays[inarr].dtype, sdfg.arrays[inarr].storage)
        output_memlet = Memlet.from_array(outarr, arr)
    else:
        inarr = in_array
        outarr = out_array

        # Convert axes to tuple
        if axis is not None and not isinstance(axis, (tuple, list)):
            axis = (axis, )
        if axis is not None:
            axis = tuple(pystr_to_symbolic(a) for a in axis)
            axis = tuple(normalize_axes(axis, len(sdfg.arrays[inarr].shape)))

        # Compute memlets
        input_subset = subsets.Range.from_array(sdfg.arrays[inarr])
        input_memlet = Memlet.simple(inarr, input_subset)
        output_subset = subsets.Range.from_array(sdfg.arrays[outarr])
        output_memlet = Memlet.simple(outarr, output_subset)

    # Create reduce subgraph
    inpnode = state.add_read(inarr)
    rednode = state.add_reduce(redfunction, axis, identity)
    outnode = state.add_write(outarr)
    state.add_nedge(inpnode, rednode, input_memlet)
    state.add_nedge(rednode, outnode, output_memlet)

    if out_array is None:
        return outarr
    else:
        return []

def _argminmax(pv: Any,
               sdfg: dace.SDFG,
               state: dace.SDFGState,
               a: str,
               axis,
               func,
               result_type=dace.int32,
               return_both=False):
    nest = NestedCall(pv, sdfg, state)

    assert func in ['min', 'max']

    if axis is None or not isinstance(axis, Integral):
        raise SyntaxError('Axis must be an int')

    a_arr = sdfg.arrays[a]

    if not 0 <= axis < len(a_arr.shape):
        raise SyntaxError("Expected 0 <= axis < len({}.shape), got {}".format(a, axis))

    reduced_shape = list(copy.deepcopy(a_arr.shape))
    reduced_shape.pop(axis)

    val_and_idx = dace.struct('_val_and_idx', idx=result_type, val=a_arr.dtype)

    # HACK: since identity cannot be specified for structs, we have to init the output array
    reduced_structs, reduced_struct_arr = sdfg.add_temp_transient(reduced_shape, val_and_idx)

    code = "__init = _val_and_idx(val={}, idx=-1)".format(
        dtypes.min_value(a_arr.dtype) if func == 'max' else dtypes.max_value(a_arr.dtype))

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_convert_".format(func),
        map_ranges={'__i%d' % i: '0:%s' % n
                    for i, n in enumerate(a_arr.shape) if i != axis},
        inputs={},
        code=code,
        outputs={
            '__init': Memlet.simple(reduced_structs,
                                    ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
        },
        external_edges=True)

    nest.add_state().add_mapped_tasklet(
        name="_arg{}_reduce_".format(func),
        map_ranges={'__i%d' % i: '0:%s' % n
                    for i, n in enumerate(a_arr.shape)},
        inputs={'__in': Memlet.simple(a, ','.join('__i%d' % i for i in range(len(a_arr.shape))))},
        code="__out = _val_and_idx(idx={}, val=__in)".format("__i%d" % axis),
        outputs={
            '__out':
            Memlet.simple(reduced_structs,
                          ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis),
                          wcr_str=("lambda x, y:"
                                   "_val_and_idx(val={}(x.val, y.val), "
                                   "idx=(y.idx if x.val {} y.val else x.idx))").format(
                                       func, '<' if func == 'max' else '>'))
        },
        external_edges=True)

    if return_both:
        outidx, outidxarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        outval, outvalarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, a_arr.dtype)

        nest.add_state().add_mapped_tasklet(
            name="_arg{}_extract_".format(func),
            map_ranges={'__i%d' % i: '0:%s' % n
                        for i, n in enumerate(a_arr.shape) if i != axis},
            inputs={
                '__in': Memlet.simple(reduced_structs,
                                      ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
            },
            code="__out_val = __in.val\n__out_idx = __in.idx",
            outputs={
                '__out_val': Memlet.simple(outval, ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis)),
                '__out_idx': Memlet.simple(outidx, ','.join('__i%d' % i for i in range(len(a_arr.shape)) if i != axis))
            },
            external_edges=True)

        return nest, (outval, outidx)

    else:
        # map to result_type
        out, outarr = sdfg.add_temp_transient(sdfg.arrays[reduced_structs].shape, result_type)
        nest(_elementwise)("lambda x: x.idx", reduced_structs, out_array=out)
        return nest, out

def _elementwise(pv: Any,
                 sdfg: dace.SDFG,
                 state: dace.SDFGState,
                 func: Union[StringLiteral, str],
                 in_array: str,
                 out_array=None):
    """
    Apply a lambda function to each element in the input.
    """

    inparr = sdfg.arrays[in_array]
    restype = sdfg.arrays[in_array].dtype

    if out_array is None:
        out_array, outarr = sdfg.add_temp_transient(inparr.shape, restype, inparr.storage)
    else:
        outarr = sdfg.arrays[out_array]

    func_ast = ast.parse(func.value if isinstance(func, StringLiteral) else func)
    try:
        lambda_ast = func_ast.body[0].value
        if len(lambda_ast.args.args) != 1:
            raise SyntaxError("Expected lambda with one arg, but {} has {}".format(func, len(lambda_ast.args.arrgs)))
        arg = lambda_ast.args.args[0].arg
        replaced_ast = astutils.ASTFindReplace({arg: '__inp'}).visit(lambda_ast.body)
        body = astutils.unparse(replaced_ast)
    except AttributeError:
        raise SyntaxError("Could not parse func {}".format(func))

    code = "__out = {}".format(body)

    num_elements = reduce(lambda x, y: x * y, inparr.shape)
    if num_elements == 1:
        inp = state.add_read(in_array)
        out = state.add_write(out_array)
        tasklet = state.add_tasklet("_elementwise_", {'__inp'}, {'__out'}, code)
        state.add_edge(inp, None, tasklet, '__inp', Memlet.from_array(in_array, inparr))
        state.add_edge(tasklet, '__out', out, None, Memlet.from_array(out_array, outarr))
    else:
        state.add_mapped_tasklet(
            name="_elementwise_",
            map_ranges={f'__i{dim}': f'0:{N}' for dim, N in enumerate(inparr.shape)},
            inputs={'__inp': Memlet.simple(in_array, ','.join([f'__i{dim}' for dim in range(len(inparr.shape))]))},
            code=code,
            outputs={'__out': Memlet.simple(out_array, ','.join([f'__i{dim}' for dim in range(len(inparr.shape))]))},
            external_edges=True)

    return out_array

