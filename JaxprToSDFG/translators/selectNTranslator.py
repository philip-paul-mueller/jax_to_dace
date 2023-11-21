"""This implements the `select_n` intrinsic.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import numpy as np
import dace
from dace import subsets
from typing import Union, Any



class SelectNTranslator(JaxIntrinsicTranslatorInterface):
    """This implements the `select_n` Jax intrinsic which acts as a generalized `where`.

    Its general notation is:
    ```
        select_n(cond, *cases)
    ```
    `cond` is either a boolean (array) in which case `*cases` represents two arrays, and the behaviour is essentially the same `where`.
    In the second mode `cond` is an integer array, whose elements are bound by `0 <= cond_i < N`, in that case `*cases` represents `N` different arrays, 
    basically we have `cases[cond_i]`, i.e. indirect indexing.

    As a simplification, the documentation makes it clear that _all_ arrays have the same shape, but scalars are allowed.

    See also:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select_n.html#jax.lax.select_n

    Notes:
        This class is based on an earlier version of the `SimpleTranslator` class.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a `briadcast_in_dim` translators
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
        return str(eqn.primitive.name) == "select_n"
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
            Jax only allows that the slicing parameters have static values.
            While the implementation could potentially handle a step size not equal than 1, Jax seems to implement that a bit different.
        """
        if(len(eqn.invars) <= 2):   # In absurd cases `==2` would make sense, but we forgot that.
            raise ValueError(f"Expexted more than two input variables but got {len(eqn.invars)}")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(not all([eqn.invars[0].aval.shape == eqn.invars[i].aval.shape  for i in range(1, len(eqn.invars)) if inVarNames[i] is not None])):
           raise ValueError(f"Expected that all the input arguments have the same shape.")
        if(len([x  for x in inVarNames if isinstance(x, str)]) == 0):
            raise ValueError(f"Only passed lterals.")
        if(not all([isinstance(inVarNames[i], str) or (inVarNames[i] is None and eqn.invars[i].aval.shape == ())  for i in range(len(inVarNames))])):
            raise ValueError(f"Found some strange input that is not handled.")
        if([eqn.invars[i]  for i in range(len(inVarNames)) if inVarNames[i] is not None][0].aval.shape != eqn.outvars[0].aval.shape):
           raise ValueError(f"Expected that input ({eqn.invars[0].aval.shape}) and output ({eqn.outvar[0].shape}) have the same shapes.")
        if(len(eqn.effects) != 0):
            raise ValueError(f"Can only handle equations without any side effects.")
        if(len(eqn.params) != 0):
            raise ValueError(f"Can only handle quations without any parameters.")
        if(inVarNames[0] is None):
            raise ValueError(f"The condition can not be a litteral")    # TODO: Optimize that case
        #

        # This works because we requiere that all non literal in/outputs have the same shape
        is_scalar = (len(eqn.outvars[0].aval.shape) == 0)

        # We only need a map range if we are not scalar
        if(not is_scalar):
            tMapRanges = [ (f'__i{dim}', f'0:{N}')  for dim, N in enumerate(eqn.invars[0].aval.shape) ]
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
            tInputs.append( (('__cond' if i == 0 else '__in{i}').format(i=i-1), iMemlet) )
        #
        if(all(x is None  for x, y in tInputs)):
            raise ValueError(f"Found only literals, which is currently not supported.")
        #

        # Generate the tasklet code
        tCode = self._writeTaskletCode(inVarNames, eqn, self._isIntDType(eqn.invars[0].aval.dtype))

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
    # end def: translateEqn


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
            intMode: bool,
    ):
        """This function generates the tasklet code based on a primitive.

        The function will also handle literal substitution.

        The mode, i.e. if bool or integer mode should be used can be controled with `intMode`.
        """
        assert len(inVarNames) >= 3

        if(not intMode):
            assert len(inVarNames) == 3
            # The order here is strange, because `False` is `0` and thus we have to select the first one.
            tCode = '__out0 = __in1 if __cond else __in0'

        else:
            # This is the integer select mode.
            nbCases = len(inVarNames) - 1       # How many cases are there?

            tCode = 'if __cond == 0:  __out0 = __in0'

            for i in range(1, nbCases):
                tCode += '\n' + 'elif __cond == {i}: __out0 = __in{i}'
            #

            # Now the undefined case
            tCode += '\n' + 'else: __out0 = math.NAN'
        #

        # Now substitute the litterals into it, the condition can be ignored.
        for i in range(1, len(inVarNames)):
            if(inVarNames[i] is not None):  continue        # We are only interessted in literals

            jaxInVar = eqn.invars[i]
            if(jaxInVar.aval.shape == ()):
                tVal = jaxInVar.val
                if(isinstance(tVal, np.ndarray)):
                    tVal = jaxInVar.val.max()                   # I do not know a better way in that case
                #
                tCode = tCode.replace(f"__in{i-1}", str(tVal))  # We have to correct for the condition.
            else:
                raise ValueError(f"Can not handle the literal case of shape: {jaxInVar.aval.shape}")
        # end for(i):

        return tCode
    # end def: _writeTaskletCode


    @staticmethod
    def _isIntDType(dType):
        """Tests if `dType` is (most likely) an integer type.
        """
        dTypeStr = str(dType)
        return any([dTypeStr.startswith(x)  for x in ['uint', 'int']])
    # end def: _isIntDType
# end class(SelectNTranslator):




