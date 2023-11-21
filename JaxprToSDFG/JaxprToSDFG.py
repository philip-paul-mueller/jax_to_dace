import numpy as np
import jax
import dace

from typing import Optional

from dace.dtypes    import DeviceType
from jax._src.core  import ClosedJaxpr, JaxprEqn, Jaxpr

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface


class JaxprToSDFG:
    """This is a simple class that allows to translate an `jaxpr` instance (a closed one) to an SDFG.

    This class should be seen as the driver, it handles tasks souch as:
    - Managing the SDFG.
    - Managing the variables and keeping track of which Jax Variables belongs to which SDFG one.

    However, it is unable to translate an equation on its own, this is delagated to a translator.
    To add one you habe to register it inside modul constant `translators.ALL_TRAFOS`.

    The idea of the transformation is quite simple.
    Since Jaxpr is essentially a list of more or less simple instructions which stores the result in one variable,
    we can just process one after the other into a "mapped tasklet" (in the simplest case) and connect them through temporaries.

    Notes:
        Equations that only have `_` as output variables are ignored.
            It seems that `grad` inserts them.

    Todo:
        Fully dynamic storage sizes or just the strides(?), i.e. make them symbols such that DaCe can play more.
    """


    ########################
    #       Initialization
    #

    def __init__(self):
        """`self` is stateless so no constructor is needed.

        The transformsers are set up in `_initEqnTranslators()`.

        """
        # We now allocate the variables of the internal state (by calling the clear function)
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

        """Contains all translators for JAX equations
        """
        self.m_eqnTranslators: Optional[list[JaxIntrinsicTranslatorInterface]] = None


        """This is the variable map, that maps the jax name to the name that is used inside the SDFG.
        You should not update this map directly instead use `_createInitialInputs()`, `_createReturnOutput()` or `_createJaxVariable()` that does this for you.
        """
        self.m_jaxNameMap: Optional[dict[str, str]] = None
    #


    def _initEqnTranslators(self, *args, **kwargs):
        """This function initializes all the transformers that are used inside `self`.
        """
        from .translators import ALL_TRAFOS

        if(self.m_eqnTranslators is not None):
            raise ValueError(f"The translators are already initialized.")
        self.m_eqnTranslators = []

        for cls in ALL_TRAFOS:
            self.m_eqnTranslators.append( cls(*args, **kwargs) )
        #
        return self
    # end def: _initEqnTranslators






    #################################################
    #   Translation interface
    #

    def transform(self,
                  jaxpr: ClosedJaxpr,
                  simplify = False,
                  auto_opt = False,
                  device: DeviceType = DeviceType.CPU,
    ) -> dace.SDFG:
        """Transforms `jaxpr` into an `SDFG`.

        By default the function just performs some verbatim style transformation.
        By setting `simplify` to `True` the function will call `simplify()` on the generated `SDFG`.
        Furthemore, by setting `auto_opt` to `True` the function will call `auto_optimize` on the `SDFG`.
        If `auto_opt` is an integer, that many times `auto_optimize` will be applied.

        Args:
            jaxpr:      The `ClosedJaxpr` instance that should be translated.
            simplify:   Apply simplify to the generated `SDFG`.
            auto_opt:   Appy `auto_optimize` on the `SDFG` before returning it.
            device:     For which device to optimize for.
        """
        import dace
        from dace import SDFG
        from dace.transformation.auto.auto_optimize import auto_optimize

        if(not jax.config.jax_enable_x64):
            raise ValueError("`x64` Support was disabled, you have to enable it by calling `jax.config.update('jax_enable_x64', True)` before anything else.")
        #

        if(auto_opt is True):
            auto_opt = 1
        elif(auto_opt is False  or  auto_opt is None):
            auto_opt = 0
        elif(isinstance(auto_opt, int)):
            if(auto_opt < 0):
                raise ValueError(f"Passed the negative value '{auto_opt}' as `auto_opt`.")
            auto_opt = int(auto_opt)
        else:
            raise TypeError(f"Does not know how to handle `{auto_opt}` ({type(auto_opt)}) passed as `auto_opt`.")
        assert isinstance(auto_opt, int) and (auto_opt >= 0)

        try:
            jaxSDFG: SDFG = self._transform(jaxpr)   # Perform the translation.
            
            if(simplify):
                jaxSDFG.simplify()
            for _ in range(auto_opt):
                jaxSDFG = auto_optimize(sdfg=jaxSDFG, device=device)
            #
            jaxSDFG.validate()      # This function throws if an error is detected.

            return jaxSDFG

        finally:
            self._clearState()
        #
    # end def: transform


    def __call__(self, jaxpr: ClosedJaxpr, *args, **kwargs) -> dace.SDFG:
        """An alias for `self.transform(jaxpr, *args, **kwargs)`.
        """
        return self.transform(jaxpr, *args, **kwargs)
    # end def: __call__




    ##########################################
    #   Variable Management
    #

    def _addArray(self, arg, isTransient=True, altName=None, forceArray = False):
        """Creates an array inside Dace for `arg` and return its name.

        Note that this function, by defaults creates transients, which is different from `SDFG.add_array()`.
        The function also distinguishes between `Scalar`s (having empty shapes) and `Array`s (non-empty shape)
        and calls `SDFG.add_scalar()` or `SDFG.add_array()` respectively.
        However, by setting `forceArray` to `True` the function will turn a `Scalar` into a one element `Array`.

        The name of the entity that is created is usually `str(arg)`, however, the function will enforce some restrictions on that.
        However, these restrictions are not applied to names passed through `altName`.

        Returns:
            The name of the array inside `self.m_sdfg`.

        Args:
            arg:            The Jax object that should be maped to dace.
            isTransient:    If a transent should be created, by default.
            forceArray:     Turn scalar in one element arrays.

        Notes:
            This function does not update the internal variable map, thus you should not use it, except you know what you are doing.
                Instead you should use `_createInitialInputs()`, `_createReturnOutput()`, `_createJaxVariable()` or `_createJaxVarList()`, that updates the map.
        """
        if isinstance(arg, jax._src.core.Var):
            propName = str(arg)             # This is the name that is _suggested_ by the convertion.
            if(altName is not None):        # Another name is passed anyway so no check is needed
                pass
            elif(propName.startswith('__')):    # names starting with `__` are DaCe internals.
                raise ValueError(f"You tried to create the variable `{propName}` which starts with two underscores, if you really want to do that use `altName`.")
            else:
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

        name      = argName
        shape     = arg.aval.shape          # Shape of the array
        offset    = None                    # i.e. no offset
        strides   = None                    # TODO(phimuell): make it fully dynamic by using symbols and let dace figuring it out.
        is_scalar = (shape == ())
        dtype     = self._translateDType(arg.aval.dtype)

        if(is_scalar and forceArray):       # "cast" the argument to an array.
            shape     = (1, )
            is_scalar = False
        #

        if(is_scalar):
            self.m_sdfg.add_scalar(name=name, dtype=dtype, transient=isTransient)
        else:
            self.m_sdfg.add_array(
                    name=name, shape=shape, strides=strides,
                    offset=offset, dtype=dtype, transient=isTransient
            )
        #
        assert name in self.m_sdfg.arrays
        return name
    # end def: _addArray


    def _addViewLike(self, likeThis, newName = None, newShape = None):
        """This function allows to create view that is is similar to `likeThis`.

        If `likeThis` is a string it is assumed to name an SDFG Array.
        If `newName` is `None` a new name will be derived.


        Args:
            likeThis:       Take anything not specified from this template.
            newName:        The new name that should be used for it.
            newShape:       The shape that should be used.

        Notes:
            The number of elements given by `likeThis` must not match `newShape`.
        """
        if isinstance(likeThis, jax._src.core.Var):
            likeThis = str(likeThis)
            assert likeThis in self.m_jaxNameMap, f"Expected to find a mapping for jax variable '{likeThis}' to a SDFG array, but it is not known."
            srcArray: dace.data.Array = self.m_sdfg.arrays[ self.m_jaxNameMap[likeThis] ]
        elif isinstance(arg, jax._src.core.Literal):
            raise NotImplementedError(f"Jax Literals are not yet implemented.")
        elif isinstance(likeThis, str):
            assert likeThis in self.m_sdfg.arrays
            srcArray: dace.data.Array = self.m_sdfg.arrays[likeThis]
        else:
            raise TypeError(f"Does not know how to handle {type(arg)}.")
        #

        if(newName is None):
            newNameTmpl = f'{likeThis}_view'
            for i in range(1, 100):
                newName = newNameTmpl + '_' + str(i)
                if(newName not in self.m_sdfg.arrays):
                    break
            else:
                raise ValueError(f"Failed to derive a new name for '{likeThis}'")
        #
        assert isinstance(newName, str)
        assert len(newName) != 0

        if(newName in self.m_sdfg.arrays):
            raise ValueError(f"Requested to generate a view with name `{newName}` but it is already taken.")
        #

        dType   = srcArray.dtype
        shape   = srcArray.shape if newShape is None else newShape
        strides = None
        offset  = None
        transient = True

        self.m_sdfg.add_view(
                name=newName,
                shape=shape,
                dtype=dType,
                strides=strides,
                offset=offset)
        return newName
    # end def: _addViewLike


    def _createInitialInputs(self, jaxpr: ClosedJaxpr):
        """Creates the initial inputs, i.e. arguments to the jax expression.

        The function will update the internal variable map.

        There is no function to create the inputs for the indivisual equations, since:
        - they are either initial input
        - literals
        - former outputs of some equations.
        """
        if(self.m_jaxNameMap is None):          # Ensure that the name map is active.
            self.m_jaxNameMap = dict()
        if(not isinstance(self.m_sdfg, dace.SDFG)):
            raise TypeError(f"The internal SDFG object is not an SDFG but '{type(self.m_sdfg)}'")
        if(len(self.m_sdfg.arg_names) != 0):
            raise ValueError(f"Expected that the argument list of the SDFG is empty but it already contains: {self.m_sdfg.arg_names}")
        #

        # We have to iterate through the non closed jaxpr, because there the names are removed.
        self.m_sdfg.arg_names = []
        for inp in jaxpr.jaxpr.invars:
            name = self._addArray(inp, isTransient=False)
            self.m_sdfg.arg_names.append(name)
            self.m_jaxNameMap[str(inp)] = name      # Add the name translation to the map.
        #
        return
    # end def: _createInitialInputs


    def _createReturnOutput(self, jaxpr: ClosedJaxpr):
        """Creates the return value statement.
        """

        # Create now the arrays that we use as output, these are the special `__return` / `__return_{IDX}` variables.
        outVarMap: dict[str, str] = {}
        for i in range(len(jaxpr.jaxpr.outvars)):
            jaxOutVar  = jaxpr.jaxpr.outvars[i]                                                         # Name of the variable inside jax/SDFG
            SDFGoutVar = ('__return' if len(jaxpr.jaxpr.outvars) == 1 else '__return_{}').format(i)     # This name will mark it as a return value.

            # Create an output array that has the same shape as `jaxOutVar` but with name `SDFGoutVar`.
            #  We have to force the creation of a container (otherwhise the code generator will safe the result in a pass by value argument).
            _ = self._addArray(jaxOutVar, isTransient=False, altName=SDFGoutVar, forceArray=True)

            assert _ == SDFGoutVar
            outVarMap[str(jaxOutVar)] = SDFGoutVar
        # end for(i):

        # Now we create the return state.
        final_state = self.m_sdfg.add_state_after(self.m_sdfgHead, label='Final_State')

        for jVar, sVar in outVarMap.items():
            jAN    = final_state.add_read(jVar)
            sAN    = final_state.add_write(sVar)
            memlet = dace.Memlet.from_array(sVar, self.getArray(jVar))
            final_state.add_edge(jAN, None, sAN, None, memlet)                      # Now we add  the connection between them
        #
        return
    # end def: _createReturnOutput


    def _createConstants(self, jaxpr: ClosedJaxpr):
        """This function creates the constants, that are named in the closure.
        """
        from copy import deepcopy
        assert self.m_jaxNameMap is not None

        # Interestingly the values and the names of the constants are kind of separated
        for cName, cValue in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            self.m_sdfg.add_constant(str(cName), deepcopy(cValue))
            self.m_jaxNameMap[cName] = cName
        #
        return
    # end def: _createConstants


    def _createJaxVarList(self, jaxVarList):
        """This function creates the listed jax variables and returns the SDFG names as a list.

        Expected input arguments are `JaxprEqn.invars` or `JaxprEqn.outvars`.
        The function will iterate through the list and return a `list` each element referes to the correspomnding SDFG variable.
        This is either a string or `None` if the variable is a literal.
        If the variable does not exists yet it will be created.
        """
        assert self.m_jaxNameMap is not None, "The variable map is not initialized."

        retList = []
        for var in jaxVarList:
            if isinstance(var, jax._src.core.Literal):
                retList.append(None)        # There is no SDFG variable for this 
            elif isinstance(var, jax._src.core.Var):
                if(str(var) in self.m_jaxNameMap):                      # The variable is known, so we just return the SDFG name.
                    retList.append( self.m_jaxNameMap[str(var)] )
                else:
                    retList.append( self._addArray(var) )               # The variable is not known, so we have to create it
                    self.m_jaxNameMap[str(var)] = retList[-1]           #  and add it to the mapping.
            else:
                raise ValueError(f"The translation process is not implemented for '{type(var)}'")
            #
        # end for(var):
        return retList
    # end def: _createJaxVarList




    ####################################
    #   Internal Translation Routines
    #

    def _transform(self, jaxpr: ClosedJaxpr) -> dace.SDFG:
        """This function does the actuall transformation, but it does not reset the internal state.

        You should not use this function directly, instead you should always call `self.transform()`.
        The reason is that `self.transform()` prepares the internal state of `self`.
        """
        if self.m_sdfg is not None:
            raise RuntimeError("Expected the that `self` is in an initial state, but it does not seem to be the case.")
        if not isinstance(jaxpr, ClosedJaxpr):
            raise TypeError(f"Expected a `jax.core.ClosedJaxp` instance but got `{type(jaxpr)}`")
        if len(jaxpr.effects) != 0:
            raise ValueError(f"Currently `Jaxpr` instances with side effects are not supported.")
        if(len(jaxpr.out_avals) == 0):
            raise ValueError(f"You have zero output variables.")
        if(not jax.config.read("jax_enable_x64")):
            raise ValueError(f"The translation only works if `jax_enable_x64` is enabled. Do it manually or use `self.transform()`!")
        #

        self.m_sdfg = dace.SDFG(name=f"jax_{id(jaxpr)}")

        # Now we create the initial state in the SDFG, which also becomes our head state.
        self.m_sdfgHead = self.m_sdfg.add_state(label="initial_state", is_start_state=True)

        # Create the translators
        self._initEqnTranslators()

        # Now we are creating the initial inputs, i.e. the ones that are named in the closure and are the arguments.
        self._createInitialInputs(jaxpr)

        # Now we are creating the constants.
        if len(jaxpr.consts) != 0:
            self._createConstants(jaxpr)
        #

        # Now transforming every equation one by one.
        for eqn in jaxpr.jaxpr.eqns:
            assert not any([str(inVar) == '_'  for inVar in eqn.invars])
            if(all([str(outVar) == '_'  for outVar in eqn.outvars])):
                assert len(eqn.effects) == 0        # This is for safe keeping, check what Jax is doing
                continue
            #
            self._translateEqn(jaxpr, eqn)
        # end for(eqn): transforming

        # Handle the output stuff
        self._createReturnOutput(jaxpr)

        return self.m_sdfg
    # end def: _transform


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
        assert len(eqn.invars)  >  0, "Expected to find at least one input variable."
        assert len(eqn.outvars) >= 1, f"Expected to find at least one output variable for equation '{str(eqn)}' but it had {len(eqn.outvars)}"
        assert all([str(out) not in self.m_jaxNameMap  for out in eqn.outvars]), f"The outputs {[str(out)  for out in eqn.outvars if str(out) in self.m_jaxNameMap]} were already created."
        assert len(eqn.effects) == 0, "This class can only handle siode efect free equations."

        # Inside this state we will add everything that is related to this equation.
        eqnState = self.m_sdfg.add_state_after(self.m_sdfgHead, label=f'{eqn.primitive.name}_{str(eqn.outvars[0])}__{id(eqn)}')

        # We now create the name list for the variables
        inVarNames  = self._createJaxVarList(eqn.invars )
        outVarNames = self._createJaxVarList(eqn.outvars)
        assert all([(o is not None) and (o in self.m_sdfg.arrays)  for o in outVarNames])

        # Now we look for the translator that can handle the primitive
        for eqnTranslator in self.m_eqnTranslators:
            if(eqnTranslator.canHandle(translator=self, inVarNames=inVarNames, outVarNames=outVarNames, eqn=eqn)):
                break   # We have found it
        else:
            raise NotImplementedError(f"Does not know how to handle primitive '{eqn.primitive.name}'.")
        #

        # Now we call the translation
        newSDFGHead = eqnTranslator.translateEqn(self, inVarNames, outVarNames, eqn, eqnState)

        if(newSDFGHead is None):
            newSDFGHead = eqnState
        elif(isinstance(newSDFGHead, dace.SDFGState)):
            pass
        else:
            raise TypeError(f"Encountered illegal types '{type(newSDFGHead)}'")
        #
        # The head has changed
        self.m_sdfgHead = newSDFGHead

        return self
    # end def: _translateEqn





    ##################################
    #   Getter
    #

    def getSDFG(self):
        """Returns the SDFG of self.
        """
        assert self.m_sdfg is not None, "The SDFG object is `None` are you sure that the translation is active."
        return self.m_sdfg
    # end def: getSDFG


    def getArray(self, name):
        """Return the array `name` inside `self`.

        Effectively a shorthand of `self.getSDFG().arrays[name]`.
        """
        return self.m_sdfg.arrays[name]
    # end def: getArray





    ##################################
    #   Misc
    #

    @staticmethod
    def _translateDType(dtype):
        """Translates the Jax datatypes into the ones used by DaCe.
        """
        nameofDType = str(dtype)

        # Make some basic checks if the datatype is okay
        if((not jax.config.read("jax_enable_x64")) and (nameofDType == 'float64')):
            raise ValueError(f"Found a `float64` type but `x64` support is disabled.")
        if(nameofDType.startswith('complex')):
            raise NotImplementedError(f"Support for complecx computation is not implemented.")
        #

        # Now extract the datatype from dace, this is extremly ugly.
        if(not hasattr(dace.dtypes, nameofDType)):
            raise TypeError(f"Could not find the type `{nameofDType}` ({type(dtype)}) in `dace.dtypes`.")
        dcDType = getattr(dace.dtypes, nameofDType)

        if(not isinstance(dcDType, dace.dtypes.typeclass)):
            raise TypeError(f"Expected that `{nameofDType}` would map to a `dace.typeclass` but it mapped to a `{type(dcDType)}`.")
        #

        return dcDType
    # end def: _translateDType

# end class(JaxprToSDFG):
   
