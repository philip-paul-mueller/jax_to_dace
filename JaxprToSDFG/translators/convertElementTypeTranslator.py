"""This file contains a converter instance.
"""

from sys import stderr

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from dace import subsets
from typing import Union


class ConvertElementTypeTranslator(JaxIntrinsicTranslatorInterface):
    """This class acts as a type caster.

    Due to the nature of SDFG and Jax this function essentially just copies the data around.
    Furthermore this function ignores the `weak_type` flag.
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
        return str(eqn.primitive.name) == "convert_element_type"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        Due to the nature of SDFG and Jax this function essentially just copies the data around.
        The actuall translation should be handled by SDFG.
        Furthermore this function ignores the `weak_type` flag.

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
        if(eqn.invars[0].aval.shape == ()):
            raise ValueError(f"The caster only works for arrays; Output variable '{outVarNames[0]}'; {eqn.invars[0].aval.dtype} -> {eqn.invars[0].aval.dtype}.")
        if(eqn.invars[0].aval.dtype == eqn.outvars[0].aval.dtype):
            print(f"The casting of variable {inVarNames[0]} to {outVarNames[0]} is unnecessary, since both are of type {eqn.outvars[0].aval.dtype}", file=stderr)
        #

        # This code is inspired from the `numpy.full` function.
        inName    = inVarNames[0]
        outName   = outVarNames[0]
        outShape  = eqn.outvars[0].aval.shape

        eqnState.add_mapped_tasklet(
            '_type_cast_',
            map_ranges={f"__i{dim}": f"0:{s}" for dim, s in enumerate(outShape)},
            inputs={'__in': dace.Memlet.simple(inName, ", ".join([f"__i{dim}" for dim in range(len(outShape))]))},
            code=f"__out = __in",
            outputs={'__out': dace.Memlet.simple(outName, ",".join([f"__i{dim}" for dim in range(len(outShape))]))},
            external_edges=True
        )
        return eqnState
    # end def: translateEqn

# end class(ConvertElementTypeTranslator):

