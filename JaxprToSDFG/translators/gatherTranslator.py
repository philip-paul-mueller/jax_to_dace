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
    """This implements A second version of the gather instruction.

    Instead of generating a map for every loop iteration it will create a map for the loop index.
    However, since we have to read symbols, we are forced to use a nested SDFG.

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

        # Every batch dimenion is associated with one dimenions of of the index array, but there is always one dimension more in the index array.
        #  This dimension contains the start indexes of the slice.
        if((len(batch_dims) + 1) != len(idxShape)):
            raise ValueError(f"There is a problem expected that the index array has a dimenision of {len(batch_dims) + 1}, but it had {len(idxShape)}")
        #

        # These are the dimensions (of the input) for which a mpa index is created.
        inpDimWithMapIdx = tuple([dim  for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims])
        assert len(inpDimWithMapIdx) == len(offset_dims)

        # We will gather implement as a copy tasklet.
        #  The reason for this is that the tasklet has to do its onw access to the array, sicne the index we have to access is inside `idxArr`.
        #  Every batch dimension has its own loop variable
        nbStateIters = idxShape[:-1]
        loopVars     = tuple([f'__i{outArrName}_gather{bd}'  for bd in batch_dims])

        idxArrySub = []         # Array to store how the access in the tasklet has to be performed.
        itSpaceVar = {}         # Stores the slice iteration variables used in each dimension.

        # Now we get the map ranges
        #  Currently we will only collect the ones associated to the slices and ignore the batch domensions.
        tMapRanges = []
        for dim, slice_size in enumerate(slice_sizes):
            if(dim not in start_index_map):
                # This dimension is fully copied
                tMapRanges.append( (f'__i{dim}', f'0:{slice_size}') )
                itSpaceVar[dim] = tMapRanges[-1][0]
                idxArrySub.append( tMapRanges[-1][0] )          #there is nthing we have to read from the index array.
                assert dim in inpDimWithMapIdx

            elif(dim in collapsed_slice_dims):
                # This dimension is only partially copied, however, since the dimension is collapsed, only a single
                #  entry is copied that comes from the index array.
                idxArrySub.append( f'__gather_{dim}' )

            else:
                # This dimension is partially copied, but since it is not colapsed, we need a map idex to copy the range.
                #  However, there is also an offset that is involved from copying.
                tMapRanges.append( (f'__i{dim}', f'0:{slice_size}') )
                idxArrySub.append( f'__gather_{dim} + {tMapRanges[-1][0]}' )
                itSpaceVar[dim] = tMapRanges[-1][0]
                assert dim in inpDimWithMapIdx
            #
            assert len(slice_sizes) == len(inpShape)
        #

        # Creating the input memlet that allows us to access the value array (from where we have to gather)
        #  from inside the tasklet and make it accessable through the name `__arr`.
        #  At this point it is not possible to tell where we access, because we are missing a index variables,
        #  they will only be accessable inide the tasklet (see below), however, we know that we will access
        #  only one element from the array.
        #  The Python Frontend does something similar, but it uses states.
        valMemlet = dace.Memlet.simple(
                inpArrName,
                ', '.join([f'0:{size}'  for size in inpShape]),
                num_accesses=1,
        )
        tInputs = [ ('__arr', valMemlet) ]

        # Now we are creating the memlts to access the index array and make them aviable inside the tasklet.
        for i, dim in enumerate(start_index_map):
            q = f'__gather_{dim}'                                   # Name of the index variable inside the tasklet.
            m = dace.Memlet.simple(idxArrName, f', '.join(loopVars) + f', {i}')  # Geting it from `idxArr[loop_var..., i]`.
            tInputs.append( (q, m) )
        #

        # Now we create the tasklet, as mentioned before it does a single access.
        tCode = '__out = __arr[' + ', '.join(idxArrySub) + ']'

        # Now the output variables.
        tOutputs_ = []
        mapIdxCnt = 0
        for dim in range(len(outShape)):
            if(dim in batch_dims):
                # This is a batch dimension, thus a loop variable is used for it.
                loopVar = loopVars[ batch_dims.index(dim) ]
                tOutputs_.append( str(loopVar) )

            else:
                # This is an offeset dimension, this means that for this dimension a map index is requiered.
                assert mapIdxCnt <= len(inpDimWithMapIdx)
                currMapDim = inpDimWithMapIdx[mapIdxCnt]
                mapIdxCnt += 1
                tOutputs_.append( itSpaceVar[currMapDim] )
            #
        #
        assert mapIdxCnt == len(inpDimWithMapIdx)
        tOutputs = [ ('__out', dace.Memlet.simple(outArrName, ', '.join(tOutputs_))) ]

        # Now we have to insert the batch index or state variable.
        tMapRanges = (
                [ (loopVars[i], f'0:{nbStateIters[i]}')  for i in range(len(batch_dims)) ]
                + tMapRanges
        )

        eqnState.add_mapped_tasklet(
            name=f"_gather_map_{outArrName}",
            map_ranges=self._listToDict(tMapRanges),
            inputs=self._listToDict(tInputs),
            code=tCode,
            outputs=self._listToDict(tOutputs),
            external_edges=True,
        )
        return eqnState
    # end def: translateEqn


    @staticmethod
    def _listToDict(inp: list[Union[tuple[None, Any], tuple[Any, Any]]]) -> dict[Any, Any]:
        """This method turns a list of pairs into a `dict`.

        However the function will filter out all entries where the key is `None`.
        """
        return {k:v  for k, v in inp if k is not None}
    # end def: _listToDict

# end class(GatherTranslator):

