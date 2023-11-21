"""Implements the device put primitive.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from dace import subsets
from typing import Union


class DevicePutTranslator(JaxIntrinsicTranslatorInterface):
    """This class acts as a put device primitive.

    Currently this function just copies data around.

    Todo:
        Make it handle all the things.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a cast translator.
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
        return str(eqn.primitive.name) == "device_put"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        Currently no device copy stuff is implement.
        However, it should be relatively simple to do since it should amount to a simple setting of parameters.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.
        """
        if(len(eqn.invars) != 1):
            raise ValueError(f"Slicing only supports one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(eqn.invars[0].aval.shape != eqn.outvars[0].aval.shape):
            raise ValueError("Expected that the input and output have the same shape.")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        #
        pDevice = eqn.params['device']
        pSrc    = eqn.params['src'   ]

        if(not ((pDevice is None) and (pSrc is None))):
            raise NotImplementedError(f"`device_put` is only implemented for `[device=None src=None]` not for `[device={pDevice} src={pSrc}]`.")
        #

        # Create the copy and let SDFG handle the casting.
        inName    = inVarNames[0]
        outName   = outVarNames[0]
        inArr     = translator.getArray(inName)
        outArr    = translator.getArray(outName)
        inSet     = subsets.Range.from_array(inArr)
        outSet    = subsets.Range.from_array(outArr)
        readNode  = eqnState.add_read(inName)
        writeNode = eqnState.add_write(outName)
        memlet    = dace.Memlet(data=inName, subset=inSet, other_subset=outSet)

        eqnState.add_nedge(readNode, writeNode, data=memlet)

        return eqnState
    # end def: translateEqn


    # end def: translateEqn

# end class(DevicePutTranslator):

