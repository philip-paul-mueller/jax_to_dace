from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from JaxprToSDFG._JaxprBaseTranslator               import JaxprBaseTranslator
from JaxprToSDFG._translatedSDFG                    import TranslatedSDFG

from jax._src.core  import ClosedJaxpr, JaxprEqn, Jaxpr
import dace
from dace import InterstateEdge
import numpy as np
from typing import Union, Any
from sys import stderr


class CondTranslator(JaxIntrinsicTranslatorInterface):
    """The `cond` primitive is essentially a if statement.

    It will translate the two branches into nested SDFG and in the parent SDFG use a selection.

    Todos:
        Handle the constant predicate case.
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
        return str(eqn.primitive.name) == "cond"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translates the `cond` primitive to its SDFG equivalent.

        The two branches of the `cond` orimitives are turned into

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.
        """
        from .common                    import AddNestedSDFG

        params   = eqn.params
        branches = params['branches']   # order is: `(False_Branch, True_Branches)`
        branches   = (('FALSE', branches[0]), ('TRUE', branches[1]))

        predVarName = inVarNames[0]        # This argument is used for the selection
        lInVarNames = inVarNames[1:]       # These arguments are used as arguments to lambda

        if(predVarName is None):
            raise NotImplementedError(f"The case of a literal predicate value is not implemented.")
        if(eqn.invars[0].aval.shape != ()):
            raise ValueError(f"The predicate variable is not a scalar.")
        if(any([x is None  for x in lInVarNames])):
            raise ValueError(f"`cond` only support non literals as arguments for its brances.")
        for br, brLambda in branches:
            if(len(brLambda.in_avals) != len(lInVarNames)):
                raise ValueError(f"The {br} lambda expected {len(brLambda.in_avals)} many arguments, but {len(lInVarNames)} were provided.")
            if(len(brLambda.out_avals) != len(outVarNames)):
                raise ValueError(f"The {br} lambda has {len(brLambda.in_avals)} many outputs, but {len(outVarNames)} were provided.")
        #

        # Get the parent SDFG.
        SDFG: dace.SDFG = translator.getSDFG()

        # Now we get a translator that will handle translate the lambda.
        lambdaTranslator: JaxprBaseTranslator = translator.clone()

        # We will now generate the two nested SDFG in their respective states.
        #  We will do that in a loop because they are so similar, we will later combine them.
        branchStates   = []
        bStateNameTmpl = 'cond_' + (predVarName if predVarName is not None else str(id(eqnState)))
        for br, brLambda in branches:
            translatedBranch: TranslatedSDFG = lambdaTranslator.translateJaxpr(brLambda, allow_empty_jaxpr=True)

            # Now we generate the input mapping, which is defined as NameOnTheOutside -> NameInsideTheNestedSDFG.
            inputNamesInside  = translatedBranch.inpNames
            outputNamesInside = translatedBranch.outNames
            inputNameMapping  = {outside: inside  for outside, inside in zip(lInVarNames,  inputNamesInside) }
            outputNameMapping = {outside: inside  for outside, inside in zip(outVarNames, outputNamesInside)}

            # This is the state into which we will add it.
            bStateName  = f"{bStateNameTmpl}__{br}"
            branchState = SDFG.add_state(label=bStateName, is_start_state=False)

            # Now lets add the nested SDFG
            nestedSDFG = AddNestedSDFG(
                    parentSDFG=SDFG,
                    translatedNestedSDFG=translatedBranch,
                    nestedState=branchState,
                    name=bStateName,
                    inputNameMapping=inputNameMapping,
                    outputNameMapping=outputNameMapping,
            )
            branchStates.append( (br, branchState) )
        # end for:

        # Since a in InterstateEdge conditions only symbols can be used, we have to turn the predicate variable into a symbol
        #  and for that we need an interstage edge, i.e. another state.
        branchCondTestState = SDFG.add_state(label=f"{bStateNameTmpl}_SWITCH", is_start_state=False)
        bSwitchNameVar = f"_{bStateNameTmpl}_var"
        SDFG.add_edge(
                src=eqnState,
                dst=branchCondTestState,
                data=InterstateEdge(assignments={bSwitchNameVar: str(predVarName)})
        )

        # This is the final guard state.
        bGuardStateName = f"_{bStateNameTmpl}_guardState"
        bGuardState     = SDFG.add_state(label=bGuardStateName, is_start_state=False)


        # Now we gonna add the two conditions.
        for br, branchState in branchStates:
            if(br == 'FALSE'):
                condition = f'{bSwitchNameVar} == 0'
            elif(br == 'TRUE'):
                condition = f'{bSwitchNameVar} != 0'
            else:
                raise NotImplementedError(f"The branch state '{br}' was not implemented.")
            #
            SDFG.add_edge(
                    src=branchCondTestState,
                    dst=branchState,
                    data=InterstateEdge(condition=condition)
            )
            SDFG.add_edge(
                    src=branchState,
                    dst=bGuardState,
                    data=InterstateEdge()
            )
        # end for:

        return bGuardState
    # end def: _translateEqn_Array

# end class(CondTranslator):


