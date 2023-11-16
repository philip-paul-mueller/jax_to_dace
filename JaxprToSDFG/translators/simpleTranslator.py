"""This module contains the simple transformator.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union, Any


class SimpleTransformator(JaxIntrinsicTranslatorInterface):
    """This class handles all the simple cases where only one tasklet is used.

    Current restrictions of the translator:
    - either 1 or 2 input values.
    - exactly one output value.
    - all non-literal operants (output and all inputs) must habe the same shape.
    - in the array case, i.e. output is an array, at least most one input can be a literal
    - in the scalar case all arguments can be litterals.

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
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if the equation can be handled by `self`.
        """
        # Only limited testing is performed here; see `translateEqn()` for more detailed explanations.
        is_scalar = (len(eqn.outvars[0].aval.shape) == 0)
        if((not (1 <= len(eqn.invars) <= 2)) or len(eqn.outvars) != 1):     # restrictions on in/output
            return False
        if((not is_scalar) and all([x is None  for x in inVarNames])):      # in array case there must be a real array among the inputs, but not in scalar case
            return False
        #

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
        """Translate a Jax-Equation into an equivalent SDFG that is created inside `eqnState`.

        Depending if a scalar or an array is handled either a mapped tasklet or a normal tasklet is created.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        """
        # In the tests below we ensures that everything has the same shape, thus to test
        # if we are scalar or not it is sufficient to check if the output is.
        is_scalar = (len(eqn.outvars[0].aval.shape) == 0)

        if(not (1 <= len(eqn.invars) <= 2)):
            raise ValueError(f"Expexted either 1 or 2 input variables but got {len(eqn.invars)}")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(outVarNames[0] is None):
            raise ValueError(f"The outut name must be a real variable.")
        if(not all([eqn.invars[0].aval.shape == eqn.invars[i].aval.shape  for i in range(1, len(eqn.invars)) if inVarNames[i] is not None])):
           raise ValueError(f"Expected that all the input arguments have the same shape.")
        if(not all([isinstance(inVarNames[i], str) or (inVarNames[i] is None and eqn.invars[i].aval.shape == ())  for i in range(len(inVarNames))])):
            raise ValueError(f"Found some strange input that is not handled.")
        if(any([I.aval.shape != eqn.outvars[0].aval.shape  for I in [jIn for jIn, iVN in zip(eqn.invars, inVarNames) if iVN is not None]])):
           raise ValueError(f"Expected that input ({eqn.invars[0].aval.shape}) and output ({eqn.outvar[0].shape}) have the same shapes.")
        if(len(eqn.effects) != 0):
            raise ValueError(f"Can only handle equations without any side effects.")
        if(len(eqn.params) != 0):
            raise ValueError(f"Can only handle quations without any parameters.")
        #

        # We only need a map range if we are not scalar
        if(not is_scalar):
            tMapRanges = [ (f'__i{dim}', f'0:{N}')  for dim, N in enumerate(eqn.invars[0].aval.shape) ]

            if(len([x  for x in inVarNames if isinstance(x, str)]) == 0):
                raise ValueError(f"Only literals as inputs is only allowed for the scalar case.")
            #
        #

        tInputs = []
        for i in range(len(eqn.invars)):
            if(inVarNames[i] is None):  # Input is a literal, so no data is needed.
                tInputs.append((None, None))        # the two `None`s are for the connector name and the memlet, they simplyfy coding bellow.
                continue
            #

            # Depending if we have a scalar or not create another memlet, they differ in what they transport
            #  The scalar one moves all, i.e. a single element and the array one is the usual map thing that iterates through everything.
            if(is_scalar):  iMemlet = dace.Memlet.from_array(inVarNames[i], translator.getSDFG().arrays[inVarNames[i]])
            else:           iMemlet = dace.Memlet.simple(inVarNames[i], ", ".join([X[0]  for X in tMapRanges]))
            tInputs.append( (f'__in{i}', iMemlet) )
        #

        # Generate the tasklet code
        tCode = self._writeTaskletCode(inVarNames, eqn)

        # As above we will now create output memlets, they follow the same logic.
        tOutputs = []
        for i in range(len(outVarNames)):
            if(is_scalar): tOutputs.append( (f'__out{i}', dace.Memlet.from_array(outVarNames[i], translator.getSDFG().arrays[outVarNames[i]])) )
            else:          tOutputs.append( (f'__out{i}', dace.Memlet.simple(outVarNames[i], ', '.join([X[0]  for X in tMapRanges]))) )
        #

        # This is the name of the tasklet and the name of the 
        tName = eqn.primitive.name

        if(is_scalar):
            # This creates the tasklet, but we have to establish the connections
            tTasklet = eqnState.add_tasklet(tName, self._listToDict(tInputs).keys(), self._listToDict(tOutputs).keys(), tCode)

            for iVar, (iConnName, iMemlet)  in filter(lambda X: X[0] is not None, zip(inVarNames, tInputs)):
                inp = eqnState.add_read(iVar)
                eqnState.add_edge(inp, None, tTasklet, iConnName, iMemlet)
            for oVar, (oConnName, oMemlet) in zip(outVarNames, tOutputs):
                out = eqnState.add_write(oVar)
                eqnState.add_edge(tTasklet, oConnName, out, None, oMemlet)
        else:
            eqnState.add_mapped_tasklet(
                name=tName,
                map_ranges=self._listToDict(tMapRanges),
                inputs=self._listToDict(tInputs),
                code=tCode,
                outputs=self._listToDict(tOutputs),
                external_edges=True,
            )
        #

        return eqnState
    # end def: _translateEqn_Array



    @staticmethod
    def _listToDict(inp: list[Union[tuple[None, Any], tuple[Any, Any]]]) -> dict[Any, Any]:
        """This method turns a list of pairs into a `dict`.

        However the function will filter out all entries where the key is `None`.
        """
        return {k:v  for k, v in inp if k is not None}
    # end def: _listToDict


    def _writeTaskletCode(
            self,
            inVarNames: list[Union[str, None]],
            eqn: JaxprEqn,
    ):
        """This function generates the tasklet code based on a primitive.

        The function will also handle literal substitution.
        """

        # Look for the template of the tasklet code.
        tName = eqn.primitive.name
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

        return tCode
    # end def: _writeTaskletCode
# end class(SimpleTransformator):







