"""This file contains everything that is related to slicing.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class SlicingTranslator(JaxIntrinsicTranslatorInterface):
    """This class handles slicing, it does this by copying the array.

    Essentially it copies a "window" which is essentailly a consecutive subset of the input array into the output.
    The window has the same size as the output array.

    Notes:
        Although the primitive would support a stepsize, i.e. `::n`, but does not use it.
            Instead it will translate it to slicing and then a gather.
            For that reason this code does not support a step size (which is called stride here).
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a slicing translators
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
        return str(eqn.primitive.name) == "slice"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        In essence it creates a copy of the array this is how slicing is implemented.
        We basically hope that DaCe is able to exploit that.
        It could also be made into a view.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        Notes:
            Jax only allows that the slicing parameters have static values.
            While the implementation could potentially handle a step size not equal than 1, Jax seems to implement that a bit different.
        """
        if(len(eqn.invars) != 1):
            raise ValueError(f"Slicing only supports one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(len(eqn.invars[0].aval.shape) != len(eqn.outvars[0].aval.shape)):
            raise ValueError("Expected that the input and output have the same numbers of dimensions.")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        #

        # Should use a map in the slicing
        use_map = False

        inAVal   = eqn.invars[0].aval            # The abstract value
        inShape  = inAVal.shape                  # The shape of the inpt value.
        outAVal  = eqn.outvars[0].aval
        outShape = outAVal.shape

        # Now we get the parameters
        sStart  = eqn.params['start_indices']                   # Fist index to slice
        sStop   = eqn.params['limit_indices']                   # Last index to slice
        sStride = eqn.params['strides']                         # The stride i.e. step index, might be `None` to indicate that it is all one.

        # We require that stride is `None`
        if(sStride is not None):
            raise NotImplementedError(f"The case of a non-None Stride in slicing is not implemented.")
        #

        if(use_map):
            tMapRanges, tOutputs_ = [], []
            for dim in range(len(outShape)):
                tMapRanges.append( (f'__i{dim}', f'0:{sStop[dim] - sStart[dim]}') )
                tOutputs_.append( tMapRanges[-1][0] )
            #

            tInputs_ = []
            for (mapItVar, _), startIdx in zip(tMapRanges, sStart):
                tInputs_.append( f'{mapItVar} + {startIdx}' )
            #

            eqnState.add_mapped_tasklet(
                f'_slicing_{str(eqn.outvars[0])}',
                map_ranges={k: v  for k, v in tMapRanges},
                inputs=dict(__in=dace.Memlet.simple(inVarNames[0], ', '.join(tInputs_))),
                code='__out = __in',
                outputs=dict(__out=dace.Memlet.simple(outVarNames[0], ', '.join(tOutputs_))),
                external_edges=True
            )

        else:
            # Use a memlet directly
            tInputs_, tOutputs_ = [], []
            for start, stop in zip(sStart, sStop):
                tInputs_.append( f'{start}:{stop}' )
                tOutputs_.append( f'0:{stop - start}' )
            #

            inAN   = eqnState.add_read(inVarNames[0])
            outAN  = eqnState.add_write(outVarNames[0])
            memlet = dace.Memlet(
                        inVarNames[0],
                        subset=', '.join(tInputs_),
                        other_subset=', '.join(tOutputs_),
            )
            eqnState.add_nedge(inAN, outAN, memlet)
        #

        return eqnState
    # end def: translateEqn

# end class(SlicingTranslator):

