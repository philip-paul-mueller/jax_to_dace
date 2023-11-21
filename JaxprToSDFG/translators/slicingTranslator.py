"""This file contains everything that is related to slicing.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class SlicingTransformator(JaxIntrinsicTranslatorInterface):
    """This class handles slicing, it does this by copying the array.

    To be specifical it handles the `slice` intrinsic, which is slicing with a step size of 1.
    If the stepsize is not one it seems to be reduced to a combination of `broadcast_in_dim` and `gather`.

    Todo:
        Implement the copy as a view.
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

        inAVal  = eqn.invars[0].aval            # The abstract value
        inShape = inAVal.shape                  # The shape of the inpt value.

        # Now we get the parameters
        sStart  = eqn.params['start_indices']                   # Fist index to slice
        sStop   = eqn.params['limit_indices']                   # Last index to slice
        sStride = eqn.params['strides']                         # The stride i.e. step index, might be `None` to indicate that it is all one.
        sStep   = tuple([1  for _ in range(len(inShape))])      # The step size which is not passed, by jax.

        # We require that stride is `None`
        if(sStride is not None):
            raise NotImplementedError(f"The case of a non-None Stride in slicing is not implemented.")
        #

        # Create access nodes for the input and output
        inAN  = eqnState.add_read(inVarNames[0])
        outAN = eqnState.add_write(outVarNames[0])

        # Now we create the Memlet that we will use for copying, we will use the `other_subset` feature for this.
        #  The doc says strings are not efficient, but they are simpler.
        memlet = dace.Memlet(
                data=inVarNames[0],
                subset=', '.join([f'{sStart[i]}:{sStop[i]}:{sStep[i]}'  for i in range(len(sStart))]),
                other_subset=', '.join([f'0:{sStop[i] - sStart[i]}'  for i in range(len(sStart))]),
        )

        # Now we add  the connection between them
        eqnState.add_nedge(inAN, outAN, memlet)

        return eqnState
    # end def: translateEqn

# end class(SlicingTransformator):

