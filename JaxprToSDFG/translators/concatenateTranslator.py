"""This file contains the concatenation operator.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class ConcatenateTranslator(JaxIntrinsicTranslatorInterface):
    """This handles concatenation of arrays.

    In essence it implement `https://www.tensorflow.org/xla/operation_semantics#concatenate`.
    Copy is implemented by several SDFGs that performs the assignment.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a cat translators
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
        return str(eqn.primitive.name) == "concatenate"
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
            Jax only allows that the slicing parameters have static values.
            While the implementation could potentially handle a step size not equal than 1, Jax seems to implement that a bit different.
        """
        outShps = [outVar.aval.shape  for outVar in eqn.outvars]
        inShps  = [inVar.aval.shape   for inVar  in eqn.invars ]

        # This is the dimension along which we perform the concatenation.
        catDim = eqn.params['dimension']

        if(any([x is None  for x in inVarNames])):
            raise ValueError(f"Can not handle literals as arguments.")
        if(not all([len(outShps[0]) == len(inVar.aval.shape)  for inVar in eqn.invars])):
            raise ValueError(f"Expected same rank for all inputs.")
        if(outShps[0][catDim] != sum([inShp[catDim]  for inShp in inShps])):
            raise ValueError(f"Something is wrong with the input, dod not found a matching amount of data.")
        # Test if all the collecting shapes are the same.


        # Counts the amount of "layers" we have copied/consumed
        allreadyCopied = 0

        for i, inVar in enumerate(inVarNames):
            iInAN  = eqnState.add_read(inVar)               # Create access nodes for the input and output
            outAN = eqnState.add_write(outVarNames[0])      # Create the output, we have to generate one for each, because otherwise DaCe will complain.

            # We have to read the whole input array.
            inSubSet = [f'0:{dimSize}'  for dimSize in inShps[i]]

            # `other_subset` defines where we want to write the data into the output array.
            #  Essentially it is the same as `subSet` with the exception of the `catDim`.
            outSubSet = inSubSet.copy()
            thisInput = inShps[i][catDim]
            outSubSet[catDim] = f'{allreadyCopied}:{allreadyCopied + thisInput}'

            iMemlet = dace.Memlet(
                    data=inVar,
                    subset=', '.join(inSubSet),
                    other_subset=', '.join(outSubSet)
            )

            # Now we add  the connection between them
            eqnState.add_nedge(iInAN, outAN, iMemlet)

            # Update the counter that we have copied
            allreadyCopied += thisInput
        #

        return eqnState
    # end def: translateEqn

# end class(ConcatenateTranslator):

