"""This module "handles" the `pjit` primitive.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
import numpy as np
from typing import Union, Any


class PJITTranslator(JaxIntrinsicTranslatorInterface):
    """`pjit` is essentially distributed stuff.

    Currently we ignore it, so this translator just generates an error and tells the user how to avoid it.
    """
    __slots__ = ()

    def __init__(self):
        """Initialize.
        """
        super().__init__()      # As requiered call the initializer of the super class
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        return str(eqn.primitive.name) == "pjit"
    # end def: canHandle


    def translateEqn(self, 
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translates the `pjit` primitive to its SDFG equivalent.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        Notes:
            Currently only generates an error.
        """

        raise NotImplementedError("The `pjit` primitive is currently not handled, to force disable it please use the `jax.disable_jit(disable=True)` contextmanager when you generating your Jaxpr.")
    # end def: _translateEqn_Array
# end class(PJITTranslator):


