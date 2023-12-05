"""This file contains everything that is related to reshape operation.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from dace import subsets
from typing import Union, Tuple


class ReshapeTranslator(JaxIntrinsicTranslatorInterface):
    """This class handles reshape operations.

    It mainly works with Memlet.
    """
    __slots__ = ()


    def __init__(self):
        """Initialization
        """
        super().__init__()      # As requiered call the initializer of the super class
        pass
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        
        return eqn.primitive.name == "reshape"
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
            raise ValueError(f"Reshape operation needs one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        #

        memlet = dace.Memlet(
                data=inVarNames[0],
                subset=', '.join([f'0:{size}' for size in eqn.invars[0].aval.shape]),
                other_subset=', '.join([f'0:{size}' for size in eqn.outvars[0].aval.shape]),
        )

        inAN  = eqnState.add_read(inVarNames[0])
        outAN = eqnState.add_write(outVarNames[0])
        eqnState.add_nedge(inAN, outAN, memlet)
        
        return eqnState
    # end def: translateEqn


# end class(DotGeneralTranslator):
