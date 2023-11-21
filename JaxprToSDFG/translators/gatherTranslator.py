"""This implements the `gather` Translator
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax.lax import GatherScatterMode
from jax._src.core import JaxprEqn
import dace
import jax
from dace import subsets, InterstateEdge
from typing import Union, Any



class GatherTranslator(JaxIntrinsicTranslatorInterface):
    """This implements the `select_n` Jax intrinsic which acts as a generalized `where`.

    Its general notation is:
    ```
        select_n(cond, *cases)
    ```
    `cond` is either a boolean (array) in which case `*cases` represents two arrays, and the behaviour is essentially the same `where`.
    In the second mode `cond` is an integer array, whose elements are bound by `0 <= cond_i < N`, in that case `*cases` represents `N` different arrays, 
    basically we have `cases[cond_i]`, i.e. indirect indexing.

    As a simplification, the documentation makes it clear that _all_ arrays have the same shape, but scalars are allowed.

    See also:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select_n.html#jax.lax.select_n

    Notes:
        This class is based on an earlier version of the `SimpleTranslator` class.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a `briadcast_in_dim` translators
        """
        super().__init__()      # As requiered call the initializer of the super class
        pass                    # There is nothing to initialize.
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        return str(eqn.primitive.name) == "gather"
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
            The implementation follows more or less `https://www.tensorflow.org/xla/operation_semantics#gather` with some adjustions from Jax.
        """
        assert(len(eqn.invars) == 2)

        outArrName = outVarNames[0]
        outArr     = eqn.outvars[0]                 # This is the array we want to write to
        outShape   = eqn.outvars[0].aval.shape
        inpArrName = inVarNames[0]
        inpArr     = eqn.invars[0]                  # The array we want to gather from
        inpShape   = inpArr.aval.shape
        idxArrName = inVarNames[1]                  # This is the array that contains the indexes, that we want to gather.
        idxArr     = eqn.invars[1]
        idxShape   = idxArr.aval.shape

        if(len(idxShape) != 2):
            raise NotImplementedError(f"Currently more than this is not supported.")
        #

        dimension_numbers            = eqn.params['dimension_numbers']
        offset_dims: tuple[int, ...] = dimension_numbers.offset_dims
        collapsed_slice_dims         = dimension_numbers.collapsed_slice_dims
        start_index_map              = dimension_numbers.start_index_map
        slice_sizes                  = eqn.params['slice_sizes']
        mode: GatherScatterMode      = eqn.params['mode']
        assert len(start_index_map) == idxShape[-1]

        if(GatherScatterMode.PROMISE_IN_BOUNDS != mode):
            raise NotImplementedError(f"The mode {mode} is not implemented.")
        #

        # Over this the copy loop goes
        batch_dims = tuple([d  for d in range(len(outShape)) if d not in offset_dims])

        # We assume this since it makes things a beet simpler.
        #  Every additional dimension would add one level to the state machine.
        if(len(batch_dims) != 1):
            raise NotImplementedError(f"The process is only implemented for the case of one batch dimensions, but you have `{len(batch_dims)}` ({batch_dims})")
        #

        # There is no single map that we can put here, instead we need a loop consisting of the state machine, with a copy map in it.
        #  Because of our assumption on the shape of `idxArr` from above, we know that we have to perfrom `idxShape[0]` many iterations.
        SDFG: dace.SDFG = translator.getSDFG()

        loop_var    = f'__gather_{outArrName}__loop_var'    # This is the state loop index of the state machine.
        nbStateIter = idxShape[0]                           #  This is the number of loops that we have to perfrom.

        if(nbStateIter <= 0):
            raise ValueError(f"In translation of `{str(eqn)}` have to perform the invalid number of `{nbStateIter}` state iterations.")
        #

        # These are the states that we need in this state machine.
        entry_state: dace.SDFGState       = SDFG.add_state(f'_gather_{outArrName}__meta_entry')
        guard_state: dace.SDFGState       = SDFG.add_state(f'_gather_{outArrName}__guard')
        loop_body_state: dace.SDFGState   = SDFG.add_state(f'_gather_{outArrName}__loop_body')
        loop_end_state: dace.SDFGState    = SDFG.add_state(f'_gather_{outArrName}__loop_end')

        # After the states we have to create the connections between them
        SDFG.add_edge(eqnState,        entry_state,     data=InterstateEdge())
        SDFG.add_edge(entry_state,     guard_state,     data=InterstateEdge(assignments={loop_var: "0"}))
        SDFG.add_edge(guard_state,     loop_end_state,  data=InterstateEdge(condition=f'{loop_var} == {nbStateIter}'))
        SDFG.add_edge(loop_body_state, guard_state,     data=InterstateEdge(assignments={loop_var: f'{loop_var} + 1'}))

        # These creates the variables `__i{dim}_gather_offset` that we use to read the index
        #  that describes where the patch in the index starts.
        assignments = {}
        for i, dim in enumerate(start_index_map):
            assignments[f'__i{dim}_gather_offset'] = f'{idxArrName}[{loop_var}, {i}]'
        #
        SDFG.add_edge(
            src=guard_state,
            dst=loop_body_state,
            data=InterstateEdge(
                condition=f'{loop_var} != {nbStateIter}',       # Check if the loop has ended.
                assignments=assignments,
            )
        )

        # This is the corrected slice size window, where we have colapsed away 1 sized dimensions.
        #  It is basically hwo much is copied in each iteration of the copy state machine.
        #  If it is the empty tuple, then we have to copy around a scalar.
        corr_slice_size = tuple([ss  for i, ss in enumerate(slice_sizes) if i not in collapsed_slice_dims])
        assert len(corr_slice_size) == len(offset_dims)
        is_scalar_patch = corr_slice_size == ()

        tMapRanges = []         # Range of the map
        tInputs_   = []         # Access elements.
        itSpaceCnt = 0          # Counter for generating iteration space.
        itSpaceVar = []
        for dim, slice_size in enumerate(slice_sizes):
            if(dim not in start_index_map):
                # This dimension is fully copied, by the map.
                tMapRanges.append( (f'__i{itSpaceCnt}', f'0:{slice_size}') )
                tInputs_.append( tMapRanges[-1][0] )
                itSpaceVar.append( tInputs_[-1] )
                itSpaceCnt += 1
            elif(dim in collapsed_slice_dims):
                # It is collapsed so we only have to copy the element that is denoted.
                #  For this we are using the index array.
                tInputs_.append( f'__i{dim}_gather_offset' )

            else:
                # The dimension is not collapsed, but there is an offset, it also opens an iteration space
                tMapRanges.append( (f'__i{itSpaceCnt}', f'0:{slice_size}') )
                tInputs_.append( tMapRanges[-1][0] + f" + __i{dim}_gather_offset" )
                itSpaceVar.append( tMapRanges[-1][0] )
                itSpaceCnt += 1
            #
        #
        assert len(itSpaceVar) == len(offset_dims)

        # Now the output variables.
        tOutputs_ = []
        for dim in range(len(outShape)):
            if(dim in offset_dims):
                iDim = offset_dims.index(dim)
                tOutputs_.append( itSpaceVar[iDim] )
            else:
                assert len(batch_dims) == 1
                tOutputs_.append( loop_var )
        #

        # The code is also very simple
        tCode = '__out0 = __in0'
        tName = f"_gather_map_{outArrName}_"

        if(is_scalar_patch):
            inAN  = loop_body_state.add_read(inpArrName)
            outAN = loop_body_state.add_write(outArrName)
            memlet = dace.Memlet(
                    data=inVarNames[0],
                    subset=', '.join(tInputs_),
                    other_subset=', '.join(tOutputs_),
            )
            loop_body_state.add_nedge(inAN, outAN, memlet)

        else:
            tInputs  = [ ('__in0',  dace.Memlet.simple(inpArrName, ', '.join(tInputs_)))   ]
            tOutputs = [ ('__out0', dace.Memlet.simple(outArrName, ', '.join(tOutputs_))) ]
            loop_body_state.add_mapped_tasklet(
                name=tName,
                map_ranges=self._listToDict(tMapRanges),
                inputs=self._listToDict(tInputs),
                code=tCode,
                outputs=self._listToDict(tOutputs),
                external_edges=True,
            )
        #

        return loop_end_state
    # end def: translateEqn


    @staticmethod
    def _listToDict(inp: list[Union[tuple[None, Any], tuple[Any, Any]]]) -> dict[Any, Any]:
        """This method turns a list of pairs into a `dict`.

        However the function will filter out all entries where the key is `None`.
        """
        return {k:v  for k, v in inp if k is not None}
    # end def: _listToDict

# end class(GatherTranslator):

