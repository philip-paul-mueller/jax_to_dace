"""This file contains everything that is related to reduction operations.
"""
from JaxprToSDFG import JaxprToSDFG
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from dace import dtypes, subsets
from typing import Union


class ReductionTranslator(JaxIntrinsicTranslatorInterface):
    """This class handles reduction operations
    """
    __slots__ = ("m_primitives", )


    def __init__(self):
        """Initialization
        """
        super().__init__()      # As requiered call the initializer of the super class
        self.m_primitives = ["reduce_min", "reduce_max", "reduce_sum", "reduce_prod", "argmin", "argmax"]
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        
        return eqn.primitive.name in self.m_primitives
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
            raise ValueError(f"Reduction operations needs one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        if eqn.primitive.name in ["argmin", "argmax"] and len(translator.getArray(inVarNames[0]).shape) == 1:
            raise ValueError(f"Dace seems to have an issue with flattened arrays for _argminmax")
        #

        redfunction = None
        identity = None
        if eqn.primitive.name == "reduce_min":
            redfunction = "lambda x, y: min(x, y)"
            identity = dtypes.max_value(translator.getArray(inVarNames[0]).dtype)
        elif eqn.primitive.name == "reduce_max":
            redfunction = "lambda x, y: max(x, y)"
            identity = dtypes.min_value(translator.getArray(inVarNames[0]).dtype)
        elif eqn.primitive.name == "reduce_sum":
            redfunction = "lambda x, y: x + y"
            identity = 0.
        elif eqn.primitive.name == "reduce_prod":
            redfunction = "lambda x, y: x * y"
            identity = 1.
        elif eqn.primitive.name == "argmin":
            redfunction = "min"
        elif eqn.primitive.name == "argmax":
            redfunction = "max"
        
        if not redfunction:
            raise NotImplementedError(f"This reduction primitive [{eqn.primitive.name}] is not implemented")

        from dace.frontend.python.replacements import _reduce, _argminmax
        if identity != None:
            _reduce(None,
                    translator.getSDFG(),
                    eqnState,
                    redfunction,
                    inVarNames[0],
                    outVarNames[0],
                    axis=eqn.params["axes"],
                    identity=identity
                    )
        else:
            nest, out = _argminmax(None,
                       translator.getSDFG(),
                       eqnState,
                       inVarNames[0],
                       eqn.params["axes"][0],
                       redfunction,
                       result_type=JaxprToSDFG._translateDType(eqn.params["index_dtype"].name)
            )
            # _argminmax generates multiple states.
            # Just keep the last one as tracked in nest obj.
            if nest:
                eqnState = nest.last_state
            
            # save the result to the output var
            output_subset = subsets.Range.from_array(translator.getArray(outVarNames[0]))
            output_memlet = dace.Memlet.simple(outVarNames[0], output_subset)
            inpnode = eqnState.add_read(out)
            outnode = eqnState.add_write(outVarNames[0])
            eqnState.add_nedge(inpnode, outnode, output_memlet)

        return eqnState
    # end def: translateEqn


# end class(DotGeneralTranslator):
