"""This module "handles" the `pjit` primitive.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from JaxprToSDFG._JaxprBaseTranslator               import JaxprBaseTranslator
from JaxprToSDFG._translatedSDFG                    import TranslatedSDFG

from jax._src.core  import ClosedJaxpr, JaxprEqn, Jaxpr
import dace
import numpy as np
from typing import Union, Any
from sys import stderr


class PJITTranslator(JaxIntrinsicTranslatorInterface):
    """`pjit` is essentially distributed stuff.

    It allows to perform an operation in a distributed stuff.
    Currently we ignore it and generate an error if this primitive is found.
    It will then output how the translation can be done without generating this primitive.
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
        """
        from jax._src.sharding_impls    import UNSPECIFIED
        from .common                    import AddNestedSDFG

        params: dict[str, Any]      = eqn.params
        lambdaJaxpr: ClosedJaxpr    = params['jaxpr']
        in_shardings                = params['in_shardings']
        out_shardings               = params['out_shardings']
        donated_invars              = params['donated_invars']
        name                        = params['name']
        keep_unused                 = params['keep_unused']
        inline                      = params['inline']

        if(not inline):
            print(f"Will inline pjit[{name}], despite it was not requested.")
        if(any(donated_invars)):
            print(f"Will not use any donated invariables.")
        if(not all([x is UNSPECIFIED  for x in in_shardings])):
            raise ValueError(f"Currently pjit only `UnspecifiedValue` as `in_shardings` is supported.")
        if(not all([x is UNSPECIFIED  for x in out_shardings])):
            raise ValueError(f"Currently pjit only `UnspecifiedValue` as `out_shardings` is supported.")
        if(len(eqn.invars) != len(lambdaJaxpr.in_avals)):
            raise ValueError(f"Lambda requiered {len(lambdaJaxpr.in_avals)} input but {len(eqn.invars)} were provided.")
        if(any([inVarName is None  for inVarName in inVarNames])):
            raise ValueError(f"`pjit` can not handle literals as arguments.")
        if(len(eqn.outvars) != len(lambdaJaxpr.out_avals)):
            raise ValueError(f"Lambda requiered {len(lambdaJaxpr.out_avals)} but {len(eqn.outvars)} were provided.")
        #

        # Now we get a translator that will handle translate the lambda.
        lambdaTranslator: JaxprBaseTranslator = translator.clone()

        # Now get the translated SDFG.
        lambdaTranslatedSDFG: TranslatedSDFG = lambdaTranslator.translateJaxpr(lambdaJaxpr)

        # Now we generate the input mapping, which is defined as NameOnTheOutside -> NameInsideTheNestedSDFG.
        inputNamesInside  = lambdaTranslatedSDFG.inpNames
        outputNamesInside = lambdaTranslatedSDFG.outNames
        inputNameMapping  = {outside: inside  for outside, inside in zip(inVarNames,  inputNamesInside) }
        outputNameMapping = {outside: inside  for outside, inside in zip(outVarNames, outputNamesInside)}

        # Now lets add the nested SDFG
        nestedSDFG = AddNestedSDFG(
                parentSDFG=translator.getSDFG(),
                translatedNestedSDFG=lambdaTranslatedSDFG,
                nestedState=eqnState,
                name=name,
                inputNameMapping=inputNameMapping,
                outputNameMapping=outputNameMapping,
        )

        return eqnState
    # end def: _translateEqn_Array
# end class(PJITTranslator):


