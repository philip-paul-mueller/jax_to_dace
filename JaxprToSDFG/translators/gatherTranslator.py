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
    """This implements the gather instruction.

    In previous versions this created a state machine.
    However, in newer version it will create multiple maps, i.e. unroll the loop statically.
    To see how this is done, check out commit `ba91e64`.

    Notes:
        https://www.tensorflow.org/xla/operation_semantics#gather
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html
    """
    __slots__ = ()


    def __init__(self):
        """Initialize.
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

        # The operation can not be done in a single map, however, it is possible to create different maps, that does the job.
        #  To keep documentation simple we will still refere to the different maps as `state`.
        SDFG: dace.SDFG = translator.getSDFG()

        # This is the number of maps that we need to create.
        nbStateIter = idxShape[0]

        if(nbStateIter <= 0):
            raise ValueError(f"In translation of `{str(eqn)}` have to perform the invalid number of `{nbStateIter}` state iterations.")
        #

        # We need a second state, since we have to define some symbols and for this we need an `InterstateEdge`
        #  The original state will contain nothing and all maps are added to the new state.
        gather_state: dace.SDFGState       = SDFG.add_state(f'_gather_{outArrName}__state')

        # This is the corrected slice size window, where we have colapsed away 1 sized dimensions.
        #  It is basically hwo much is copied in each iteration of the copy state machine (actually parallel maps).
        #  If it is the empty tuple, then we have to copy around a scalar.
        corr_slice_size = tuple([ss  for i, ss in enumerate(slice_sizes) if i not in collapsed_slice_dims])
        assert len(corr_slice_size) == len(offset_dims)
        is_scalar_patch = corr_slice_size == ()

        # These are all the variables, and their definitions we need in the state machine.
        #  These are the indexes that we read from the index array.
        assignments: dict[str, str] = {}

        # We will now statically unrolle the state machine.
        for loop_var in range(nbStateIter):
            # Looking up the indexes that we need.
            stateVars: dict[int, str] = {}
            for i, dim in enumerate(start_index_map):
                stateVar = f'__gather_{outArrName}_s{loop_var}_i{dim}'          # This is the variable name that tell us what start index we have to use for dimension `dim` in state `loop_var`.
                stateVars[dim] = stateVar
                assignments[stateVar] = f'{idxArrName}[{loop_var}, {i}]'        # Now store assignement for the later creation of teh interstate edge.
                del stateVar
            #

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
                    tInputs_.append( stateVars[dim] )

                else:
                    # The dimension is not collapsed, but there is an offset, it also opens an iteration space
                    tMapRanges.append( (f'__i{itSpaceCnt}', f'0:{slice_size}') )
                    tInputs_.append( tMapRanges[-1][0] + " + " + stateVars[dim] )
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
                    tOutputs_.append( str(loop_var) )
            #

            # The code is also very simple
            tCode = '__out0 = __in0'
            tName = f"_gather_map_{outArrName}_state{loop_var}_"

            if(is_scalar_patch):
                inAN  = gather_state.add_read(inpArrName)
                outAN = gather_state.add_write(outArrName)
                memlet = dace.Memlet(
                        data=inVarNames[0],
                        subset=', '.join(tInputs_),
                        other_subset=', '.join(tOutputs_),
                )
                gather_state.add_nedge(inAN, outAN, memlet)

            else:
                tInputs  = [ ('__in0',  dace.Memlet.simple(inpArrName, ', '.join(tInputs_)))   ]
                tOutputs = [ ('__out0', dace.Memlet.simple(outArrName, ', '.join(tOutputs_))) ]
                gather_state.add_mapped_tasklet(
                    name=tName,
                    map_ranges=self._listToDict(tMapRanges),
                    inputs=self._listToDict(tInputs),
                    code=tCode,
                    outputs=self._listToDict(tOutputs),
                    external_edges=True,
                )
            # end if: is scalar or not.
        # end for(loop_var):

        # Now we create connect the two states together.
        SDFG.add_edge(
            src=eqnState,
            dst=gather_state,
            data=InterstateEdge(assignments=assignments)
        )

        return gather_state 
    # end def: translateEqn


    @staticmethod
    def _listToDict(inp: list[Union[tuple[None, Any], tuple[Any, Any]]]) -> dict[Any, Any]:
        """This method turns a list of pairs into a `dict`.

        However the function will filter out all entries where the key is `None`.
        """
        return {k:v  for k, v in inp if k is not None}
    # end def: _listToDict

# end class(GatherTranslator):

