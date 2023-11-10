import numpy as np
import jax
import dace

from typing import Optional

from jax._src.core import ClosedJaxpr, JaxprEqn, Jaxpr


class JaxprToSDFG:
    """This is a simple class that allows to translate an `jaxpr` instance (a closed one) to an SDFG.

    The implementation of this class does not handle the case of dynamic translation.
    It is a stateless object and all internal members are cleared at the end.

    The translation handles the `return` statement in a very strange way.
    Instead of returning it, a special argument called `_out` is generated and will be used to store the value.

    Todo:
    - Allow fir a real return value.
    - there is some issue with the datatype.
    - Litterals are not handled at all.
    - Fully dynamic storage sizes.
    """

    def __init__(self):
        """`self` is stateless so no constructor is needed.
        """
        # We now allocate the variables of the internal state (by calling the clear function)
        self._clearState()
    #


    def __call__(self, jaxpr: ClosedJaxpr) -> dace.SDFG:
        """An alias for `self.transform(jaxpr)`.
        """
        return self.transform(jaxpr)
    #


    def transform(self, jaxpr: ClosedJaxpr) -> dace.SDFG:
        """Transforms the `jaxpr` into an SDFG and returns it.

        The idea of the transformation is quite simple.
        Since Jaxpr is essentially a list of more or less simple instructions we can just loop through them and transform them, each of them translates to a single map.
        """
        try:
            return self._transform(jaxpr)
        finally:
            self._clearState()
    #


    def _clearState(self):
        """Sets all internal variables to `None`.
        """

        """The SDFG object that we are currently constructing.
        """
        self.m_sdfg: Optional[dace.SDFG]  = None

        """This is the HEAD SDFG state, i.e. the last state in which we translated an equation.
        """
        self.m_sdfgHead: Optional[dace.SDFGState] = None

        """Maps sizes (literals) to their respective symbol names in the SDFG.
        Since SDFG has symbolic sizes but Jax has concrete sizes, we use this maping to ensure that all sizes have the same symbol.
        """
        #self.m_shapeSizeSymbols: Optional[dict[int, str]] = None
    #


    def _transform(self, jaxpr: ClosedJaxpr) -> dace.SDFG:
        """This function does the actuall transformation, but it does not reset the internal state.
        """
        if self.m_sdfg is not None:
            raise RuntimeError("Expected the that `self` is in an initial state, but it does not seem to be the case.")
        if not isinstance(jaxpr, ClosedJaxpr):
            raise TypeError(f"Expected a `jax.core.ClosedJaxp` instance but got `{type(jaxpr)}`")
        if len(jaxpr.effects) != 0:
            raise ValueError(f"Currently `Jaxpr` instances with side effects are not supported.")
        if len(jaxpr.literals) != 0:
            raise ValueError(f"Currently `Jaxpr` instances with literals are not supported.")
        if len(jaxpr.consts) != 0:
            raise ValueError(f"Currently `Jaxpr` instances with constants are not supported.")
        if(len(jaxpr.out_avals) > 1):
            raise ValueError(f"Currently only one output value is supported, but you passed {len(jaxpr.out_avals)}")
        #

        self.m_sdfg = dace.SDFG(name=f"jax_{id(jaxpr)}")

        # Now we create the initial state in the SDFG, which also becomes our head state.
        self.m_sdfgHead = self.m_sdfg.add_state(label="initial_state", is_start_state=True)

        # Now we are creating the inputs
        self._createInputs(jaxpr)

        # Now transforming every equation one by one.
        for eqn in jaxpr.jaxpr.eqns:
            self._translateEqn(jaxpr, eqn)
        #

        # Handle the output stuff
        self._createOutputs(jaxpr)

        return self.m_sdfg
    #


    def _createInputs(self, jaxpr: ClosedJaxpr):
        """Creates the initial inputs.
        """
        # We have to iterate through the non closed jaxpr, because there the names are removed.
        for inp in jaxpr.jaxpr.invars:
            _ = self._addArray(inp, isTransient=False)
        #
        return
    #


    def _createOutputs(self, jaxpr: ClosedJaxpr):
        """Creates the return value statement.

        However, currently it is a big fat hack, since it creates an artificial argument and copies the arguments back.
        """
        if(len(jaxpr.out_avals) == 0):
            return
        elif(len(jaxpr.out_avals) == 1):
            pass
        else:
            raise ValueError(f"Expected at most one return argument, but found {len(jaxpr.out_avals)}")
        #

        # Create now the array that we use as output.
        outVar  = jaxpr.jaxpr.outvars[0]
        outAVal = outVar.aval
        self._addArray(outVar, isTransient=False, altName="_out")

        # This is the final state that we use
        final_state = self.m_sdfg.add_state_after(self.m_sdfgHead, label='Final_State')

        final_state.add_mapped_tasklet(
                name="RETURN",
                map_ranges={f'__i{dim}': f'0:{N}'  for dim, N in enumerate(outAVal.shape)},
                inputs=dict(__in=dace.Memlet.simple(str(outVar), ", ".join([f'__i{dim}'  for dim in range(len(outAVal.shape))]))),
                code="__out = __in",
                outputs=dict(__out=dace.Memlet.simple("_out", ", ".join([f'__i{dim}'  for dim in range(len(outAVal.shape))]))),
                external_edges=True,
        )

        return
    # end def: _createOutput


    def _addArray(self, arg, isTransient=True, altName=None):
        """Creates an array inside Dace for `arg` and return its name.

        Note that this function, by defaults creates transients.
        This is different from the `add_array()` function of DaCe.

        Args:
            arg:            The Jax object that should be maped to dace.
            isTransient:    If a transent should be created, by default.
        """
        if isinstance(arg, jax._src.core.Var):
            pass
        elif isinstance(arg, jax._src.core.Literal):
            raise NotImplementedError(f"Jax Literals are not yet implemented.")
        else:
            raise TypeError(f"Does not know how to handle {type(arg)}.")
        #

        argName = str(arg) if altName is None else str(altName)     # Ensure that we have a string.
        if argName in self.m_sdfg.arrays:
            raise ValueError(f"The variable `{str(arg)}` is already recorded in the SDFG.")
        if(len(argName) == 0):
            raise ValueError(f"Got an empty name.")
        elif(argName[0].isdigit()):
            raise ValueError(f"Requested to create the array '{arg}', is ilegal since it starts with a digit.")
        elif(any([x.isspace()  for x in argName])):
            raise ValueError(f"The name of the array, '{arg}', to create contained a space!")
        #

        name    = argName
        shape   = arg.aval.shape
        offset  = None          # i.e. no offset
        strides = None          # i.e. C-Layout
                                # TODO(phimuell): make it fully dynamic by using symbols and let dace figuring it out.
        dtype   = self._translateDType(arg.aval.dtype)
        self.m_sdfg.add_array(
                name=name, shape=shape, strides=strides,
                offset=offset, dtype=dtype, transient=isTransient
        )
        assert name in self.m_sdfg.arrays
        return name
    # end def: _addArray


    def _translateEqn(self,
            closedJaxp: ClosedJaxpr,
            eqn: JaxprEqn,
    ):
        """This function translates the equation (statement) to an SDFG state.

        The new state will be added directly after the current head.
        It is important that the non closed equation is passed to this function, while the closed jaxpr is passed.
        This function will also modify the current head.
        """
        assert isinstance(eqn, jax._src.core.JaxprEqn)
        assert all([str(out) not in self.m_sdfg.arrays  for out in eqn.outvars])
        assert all([str(inp)     in self.m_sdfg.arrays  for inp in eqn.invars ]), f"Expected to find input '{[ str(inp)  for inp in eqn.invars if str(inp) not in self.m_sdfg.arrays]}'"
        assert len(eqn.invars)  >  0
        assert len(eqn.outvars) == 1, f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}"
        assert all([eqn.outvars[0].aval.shape == inp.aval.shape  for inp in eqn.invars]), f"Found different shapes."
        assert len(eqn.effects) == 0

        # Inside this state we will add everything that is related to this equation.
        eqnState = self.m_sdfg.add_state_after(self.m_sdfgHead, label=f'{eqn.primitive.name}_{id(eqn)}')

        # Now we create the variables for the output arrays
        outVars = []
        for out in eqn.outvars:
            outVar = self._addArray(out)
            outVars.append(outVar)
        #
        pName = eqn.primitive.name

        if pName == "__my_special_name":
            raise NotImplementedError(f"Does not know how to handle primitive '{pName}'.")
        else:
            self._handleSimpleCase(closedJaxp=closedJaxp, eqn=eqn, eqnState=eqnState, outVars=outVars)
        #

        # The head has changed
        self.m_sdfgHead = eqnState

        return self
    # end def: _translateEqn


    def _handleSimpleCase(self,
            closedJaxp: ClosedJaxpr,
            eqn: JaxprEqn,
            eqnState: dace.SDFGState,
            outVars: list[str],

    ):
        """This function handles the most simple cases, where basically a mapped tasklet can be used for the translation.

        Args:
            closedJaxp:         The `Jaxpr` closed variable, i.e. the object that should be translated.
            eqn:                The equation (statement) that is currently translated.
            eqnState:           The `SDFGState` into which we will generate the translated version.
            outVars:            List of all arrays into which we write the result.

        This function assumes the following:
        - Simple arethmetic operation `+`, `-`, `*`, `/` or `-` (unary).
        - Mathematicla operation, such as `sin`, `cos`, ...
        - One single output parameter.
        - Essentially an element whise operation, thus shape of input equal the one of the output.
        """

        if(not (1 <= len(eqn.invars) <= 2)):
            raise ValueError(f"Expexted either 1 or 2 input variables but got {len(eqn.invars)}")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(not all([eqn.invars[0].aval.shape == eqn.invars[i].aval.shape  for i in range(1, len(eqn.invars))])):
           raise ValueError(f"Expected that the input arguments have the same shape.")
        if(eqn.invars[0].aval.shape != eqn.outvars[0].aval.shape):
           raise ValueError(f"Expected that input ({eqn.invars[0].aval.shape}) and output ({eqn.outvar[0].shape}) have the same shapes.")
        assert all([outVar in self.m_sdfg.arrays  for outVar in outVars])

        unarryOps = {
                "???": "__out = +(__in0)",
                "??": "__out = -(__in0)",
                "sin":      "__out = sin(__in0)",
                "cos":      "__out = cos(__in0)",
        }
        binarryOps = {
                "add":      "__out = (__in0)+(__in1)",
                "sub":      "__out = (__in0)-(__in1)",
                "mul":      "__out = (__in0)*(__in1)",
                "div":      "__out = (__in0)/(__in1)",
                "pow":      "__out = (__in0)**(__in1)",
        }

        # We will now create a mapped tasklet that will do all the calculations.
        tName = eqn.primitive.name
        tMapRanges = {f'__i{dim}': f'0:{N}'  for dim, N in enumerate(eqn.invars[0].aval.shape)}

        tInpNames = [ str(x)  for x in eqn.invars ]
        tInputs = {}
        for inputI in range(len(eqn.invars)):
            tInputs[f'__in{inputI}'] = dace.Memlet.simple(tInpNames[inputI], ", ".join([f'__i{dim}'  for dim in range(len(eqn.invars[inputI].aval.shape))]))
        #

        tCode = None
        for M in [unarryOps, binarryOps]:
            if(tName in M):
                tCode = M[tName]
                break
        else:
            raise ValueError(f"Does not know how to translate primitive '{tName}'")
        #

        tOutName = str(eqn.outvars[0])
        tOutputs = dict(__out=dace.Memlet.simple(tOutName, ', '.join([f'__i{dim}'  for dim in range(len(eqn.outvars[0].aval.shape))])))

        eqnState.add_mapped_tasklet(
                name=tName,
                map_ranges=tMapRanges,
                inputs=tInputs,
                code=tCode,
                outputs=tOutputs,
                external_edges=True,
        )

        return self
    # end def: _handleSimpleCase


    @staticmethod
    def _translateDType(dtype):
        """Translate some special interest dtypes into others more usefull types.
        """
        # TODO(phimuell):
        #  Why does `return dtype` does not work it is essentially `arg.aval.dtype`?
        return np.float64
    # end def: _translateDType

# end class(JaxprToSDFG):
   
