"""This file contains everything that is related to slicing.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class DotGeneralTranslator(JaxIntrinsicTranslatorInterface):
    """This class handles dot_general
    """
    __slots__ = ()


    def __init__(self):
        """Initialization
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
        return str(eqn.primitive.name) == "dot_general"
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
        if(len(eqn.invars) != 2):
            raise ValueError(f"Dot General only supports two input arguments.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        #

        output_is_scalar = (len(eqn.outvars[0].aval.shape) == 0)
        lhs_shape = eqn.invars[0].aval.shape
        rhs_shape = eqn.invars[1].aval.shape

        if len(lhs_shape) == 1 and len(rhs_shape) == 1:
            from dace.frontend.python.replacements import dot
            dot(None,
                translator.m_sdfg,
                eqnState,
                inVarNames[0],
                inVarNames[1],
                outVarNames[0],
                )
        elif len(lhs_shape) == 2 or len(rhs_shape) == 2:
            from dace.libraries.blas.nodes import MatMul
            libnode = MatMul('_MatMult_')
            eqnState.add_node(libnode)
            
            gX = eqnState.add_read(inVarNames[0])
            gY = eqnState.add_read(inVarNames[1])
            gZ = eqnState.add_write(outVarNames[0])

            eqnState.add_edge(gX, None, libnode, '_a', dace.memlet.Memlet.from_array(gX.data, translator.m_sdfg.arrays[inVarNames[0]]))
            eqnState.add_edge(gY, None, libnode, '_b', dace.memlet.Memlet.from_array(gY.data, translator.m_sdfg.arrays[inVarNames[1]]))
            eqnState.add_edge(libnode, '_c', gZ, None, dace.memlet.Memlet.from_array(gZ.data, translator.m_sdfg.arrays[outVarNames[0]]))
        else:
            raise NotImplementedError

        return eqnState
    # end def: translateEqn


# end class(DotGeneralTranslator):

