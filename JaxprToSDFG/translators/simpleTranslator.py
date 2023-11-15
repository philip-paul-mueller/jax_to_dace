"""This module contains the simple transformator.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class SimpleTransformator(JaxIntrinsicTranslatorInterface):
    """This class handles all the simple cases where only one tasklet is used.

    Examples for simple operations are:
    - Arethmetic operations.
    - Mathematical Functions.

    However, `reducing` operations are not included, such as `numpy.amin`.
    """
    __slots__ = ("m_unarryOps", "m_binarryOps")


    def __init__(self):
        """Initializes a simple translators.
        """
        super().__init__()      # As requiered call the initializer of the super class

        # We will now create maps, that maps the name of a primitive to a certain tasklet code.
        self.m_unarryOps = {
                "pos":      "__out0 = +(__in0)",
                "neg":       "__out0 = -(__in0)",

                "floor":    "__out0 = floor(__in0)",
                "ceil":     "__out0 = ceil(__in0)",
                "round":    "__out0 = round(__in0)",
                "abs":      "__out0 = abs(__in0)",
                "sign":     "__out0 = sign(__in0)",

                "sqrt":     "__out0 = sqrt(__in0)",

                "log":      "__out0 = log(__in0)",
                "exp":      "__out0 = exp(__in0)",

                "sin":      "__out0 = sin(__in0)",
                "asin":     "__out0 = asin(__in0)",
                "cos":      "__out0 = cos(__in0)",
                "acos":     "__out0 = acos(__in0)",
                "tan":      "__out0 = tan(__in0)",
                "atan":     "__out0 = atan(__in0)",
                "tanh":     "__out0 = tanh(__in0)",
        }
        self.m_binarryOps = {
                "add":      "__out0 = (__in0)+(__in1)",
                "sub":      "__out0 = (__in0)-(__in1)",
                "mul":      "__out0 = (__in0)*(__in1)",
                "div":      "__out0 = (__in0)/(__in1)",

                "rem":      "__out0 = (__in0)%(__in1)",

                "pow":      "__out0 = (__in0)**(__in1)",
                "ipow":     "__out0 = (__in0)**(int(__in1))",

                "min":      "__out0 = min(__in0, __in1)",
                "max":      "__out0 = max(__in0, __in1)",
        }
    # end def: __init__


    def canHandle(self,
                  translator,
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """

        if(len(eqn.invars) == 1):
            return str(eqn.primitive.name) in self.m_unarryOps
        elif(len(eqn.invars) == 2):
            return str(eqn.primitive.name) in self.m_binarryOps
        #
        return False
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        Essentially this is a wrapper arround `add_mapped_tasklet()`.
        This function is able to handle `None` values inside `inVarNames` if they are scalars.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        """
        if(not (1 <= len(eqn.invars) <= 2)):
            raise ValueError(f"Expexted either 1 or 2 input variables but got {len(eqn.invars)}")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(not all([eqn.invars[0].aval.shape == eqn.invars[i].aval.shape  for i in range(1, len(eqn.invars)) if inVarNames[i] is not None])):
           raise ValueError(f"Expected that all the input arguments have the same shape.")
        if(len([isinstance(inVarNames[i], str)  for i in range(len(inVarNames))]) == 0):
            raise ValueError(f"Only passed lterals.")
        if(not all([isinstance(inVarNames[i], str) or (inVarNames[i] is None and eqn.invars[i].aval.shape == ())  for i in range(len(inVarNames))])):
            raise ValueError(f"Found some strange input that is not handled.")
        if([eqn.invars[i]  for i in range(len(inVarNames)) if inVarNames[i] is not None][0].aval.shape != eqn.outvars[0].aval.shape):
           raise ValueError(f"Expected that input ({eqn.invars[0].aval.shape}) and output ({eqn.outvar[0].shape}) have the same shapes.")
        if(len(eqn.effects) != 0):
            raise ValueError(f"Can only handle equations without any side effects.")
        if(len(eqn.params) != 0):
            raise ValueError(f"Can only handle quations without any parameters.")
        #

        # We will now create a mapped tasklet that will do all the calculations.
        tName = eqn.primitive.name
        tMapRanges = {f'__i{dim}': f'0:{N}'  for dim, N in enumerate(eqn.invars[0].aval.shape)}

        # Creates the input connectors and the associated memlets
        tInputs = {}
        for i in range(len(eqn.invars)):
            if(inVarNames[i] is None):  continue        # If the input is a literal, then we do not have to create a memlet.
            tInputs[f'__in{i}'] = dace.Memlet.simple(inVarNames[i], ", ".join([f'__i{dim}'  for dim in range(len(eqn.invars[i].aval.shape))]))
        #
        if(len(tInputs) == 0):
            raise ValueError(f"Found only literals, which is currently not supported.")
        #

        # Now we generate the code that we will feed into the tasklet afterwards
        #  We do not handle literals yet.
        for M in [self.m_unarryOps, self.m_binarryOps]:
            if(tName in M):
                tCode = M[tName]
                break
        else:
            raise ValueError(f"Does not know how to translate primitive '{tName}'")
        #

        # We now handle litterals, we do this with simple text substitution in the code, since there is no input connector.
        for i in range(len(inVarNames)):
            if(inVarNames[i] is not None):  continue        # We are only interessted in literals

            jaxInVar = eqn.invars[i]
            if(jaxInVar.aval.shape == ()):
                tCode = tCode.replace(f"__in{i}", str(jaxInVar.val.max()))       # I do not know a better way in that case
            else:
                raise ValueError(f"Can not handle the literal case of shape: {jaxInVar.aval.shape}")
        # end for(i):

        # Now create the output connector (`__out`) and the memlet
        tOutputs = {}
        for i in range(len(outVarNames)):
            tOutputs[f'__out{i}'] = dace.Memlet.simple(outVarNames[i], ', '.join([f'__i{dim}'  for dim in range(len(eqn.outvars[i].aval.shape))]))
        #

        eqnState.add_mapped_tasklet(
                name=tName,
                map_ranges=tMapRanges,
                inputs=tInputs,
                code=tCode,
                outputs=tOutputs,
                external_edges=True,
        )

        return eqnState
    # end def: translateEqn

# end class(SimpleTransformator):







