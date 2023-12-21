import numpy as np
import jax
import dace
import sys

from typing import Optional

from dace.dtypes    import DeviceType
from jax._src.core  import ClosedJaxpr, JaxprEqn, Jaxpr

from JaxprToSDFG._translatedSDFG                 import TranslatedSDFG


class JaxprBaseTranslator:
    """This is a simple class that allows to translate an `jaxpr` instance (a closed one) to an SDFG.

    This class does not generate a fully functional SDFG, it just takes a `jaxpr` returns an SDFG.
    It is intended as a desposable object for a single translation.

    This class should be seen as the driver, it handles tasks souch as:
    - Managing the SDFG.
    - Managing the variables and keeping track of which Jax Variables belongs to which SDFG one.

    However, it is unable to translate an equation on its own, this is delagated to a translator.
    To add one you habe to register it inside modul constant `translators.ALL_TRAFOS`.

    The idea of the transformation is quite simple.
    Since Jaxpr is essentially a list of more or less simple instructions which stores the result in one variable,
    we can just process one after the other into a "mapped tasklet" (in the simplest case) and connect them through temporaries.

    It is important that the translator will not insert any code that DaCe will turn into a return statement.

    Notes:
        Equations that only have `_` as output variables are ignored.
            It seems that `grad` inserts them.
        If you start reading the code start at the `transform()` function.
        The class only has a (allocated) members during the transformatiosn, otherwhise they are `None`.
        If a translation failes the internal state is not deallocated, before you can use the object again,
            you have to call `_clearState()` manually.

    Todo:
        Shape might be constant but stride not necessaraly, make it possible to use different strides.
    """


    ########################
    #       Initialization
    #

    def __init__(
            self
    ):
        """`self` is generally stateless, but maintains some variables during translation.

        """
        super().__init__()
        self._clearState()                                  # We now allocate the variables of the internal state
    # end def: __init__


    def clone(self):
        """Return an exact clone of `self`.

        The returned object will be in a state as `self` was right after the `self.__init__()` has finished.
        The main intention of this function is to allow recursion of the translator.

        Notes:
            It is important that this guarantee would also hold if `__init__()` would accpet arguments.
        """
        return JaxprBaseTranslator()
    # end def: clone


    @staticmethod
    def rebuildFrom(translatedSDFG: TranslatedSDFG) -> 'JaxprBaseTranslator':
        """Builds a `JaxprBaseTranslator` with the given state, that is stored inside a `translatedSDFG`.

        Args:
            translatedSDFG:     The result of a previous call to `JaxprBaseTranslator._translateJaxpr()`.

        Notes:
            This function will construct an `JaxprBaseTranslator` and allocate all necessary function.
            This is a static function, there to load an already existing `JaxprBaseTranslator` you can use the `_load_state()` function.
        """
        self = JaxprBaseTranslator()
        self._load_state(translatedSDFG)
        return self
    # end def: _restore_from



    ####################################
    #   Translation Routines
    #

    def translateJaxpr(
            self,
            jaxpr: ClosedJaxpr,
            *,
            sclar_as_array: bool = False,
            device: dace.DeviceType = dace.DeviceType.CPU,
            allow_empty_jaxpr: bool = False,
    ) -> TranslatedSDFG:
        """This funcunction generates an `SDFG` out of an `Jaxpr`.

        The function will do the following:
        - Allocate the internal state of `self`.
        - Forward the call to `self._translateJaxprInternal()`, see there for more information.
        - Clean up `self`
        - Return the result.

        The function will return a `TranslatedSDFG` (a dataclass) instances, with the following fields.
        - `sdfg` the `SDFG` object that was created.
        - `statrtState` the first state in the `SDFG` state machine.
        - `finState` the last state in the state machine.
        - `jaxNameMap` a `dict` that maps every JAX name to its corresponding SDFG variable name.
        - `inpNames` a `list` of the `SDFG` variables that are used as input.
        - `outNames` a `list` of the `SDFG` variables that are used as output.

        The lists `{inp, out}Names` have the same order as their corresponding lists in the `Jaxpr`.

        Args:
            sclar_as_array:     Translate scalar _input_ arguments to arrays.
            allow_empty_jaxpr:  Allow empty jaxpr instances.

        Notes:
            It is important that the returned `SDFG` does not contain any state that needed for returning values.
            You will most likely never need to set `sclar_as_array` to `True`.
            All variables created are transients, it might be needed to turn them into globals.
        """
        if(self._isStateAllocated()):
            raise RuntimeError("`self` is already allocated.")
        if((len(jaxpr.jaxpr.eqns) == 0) and (not allow_empty_jaxpr)):
            raise ValueError(f"Your `Jaxpr` has zero equations.")
        #

        # Allocate the internal state of self.
        self._allocState()

        # Perform the transformation.
        self._translateJaxprInternal(jaxpr, inp_sclar_as_array=sclar_as_array, device=device)

        # Now export it and clean `self`.
        #  This indirect and clumsy way allows us to avoid some 'uninitialized transient' warnings.
        retVal: TranslatedSDFG = self._export_state(doCleaning=False, jaxpr=jaxpr)
        retVal.validate()
        self._clearState()

        return retVal
    # end def: _translateJaxpr



    ########################
    #       Internal Initialization
    #

    def _clearState(self):
        """Sets all internal variables to `None`.
        """

        """The SDFG object that we are currently constructing.
        """
        self.__m_sdfg: Optional[dace.SDFG]  = None

        """This is the HEAD SDFG state, i.e. the last state in which we translated an equation.
        """
        self.__m_sdfgHead: Optional[dace.SDFGState] = None

        """This is the beginning of the SDFG, i.e. the original SDFG HEAD.
        """
        self.__m_sdfgInitState: Optional[dace.SDFGState] = None

        """Contains all translators for JAX equations
        """
        self.__m_eqnTranslators: Optional[list[JaxIntrinsicTranslatorInterface]] = None

        """This is the variable map, that maps the jax name to the name that is used inside the SDFG.
        You should not update this map directly instead use `_createInitialInputs()`, `_createReturnOutput()` or `_createJaxVariable()` that does this for you.
        """
        self.__m_jaxNameMap: Optional[dict[str, str]] = None
    #


    def _allocState(self):
        """Allocate the internal state of `self`.

        This will create the necessary variables and create the equation translators.

        Notes:
            The variables that are related to the SDFG are not initialized.
        """
        if(self._isStateAllocated()):
            raise ValueError("Expected that the internal state of the translator is unallocated.")
        #
        self.__m_jaxNameMap: dict[str, str] = dict()
        self._initEqnTranslators()

        return self
    # end def: _allocState


    def _initEqnTranslators(self):
        """This function initializes all the transformers that are used inside `self`.

        Calling this function is dangerous since it depends on some half iniiated state.
        If you think you must allocate it then call `_allocState()` instead.
        """
        from .translators import ALL_TRAFOS

        if(self.__m_eqnTranslators is not None):
            raise ValueError(f"The translators are already initialized.")
        assert self.__m_jaxNameMap is not None
        self.__m_eqnTranslators: list[JaxIntrinsicTranslatorInterface] = []

        for cls in ALL_TRAFOS:
            self.__m_eqnTranslators.append( cls() )
        #
        return self
    # end def: _initEqnTranslators


    def _isStateAllocated(self, beStrict: bool = True):
        """Test if `self` has an allocated state.
        """
        isAllocated = [x is not None  for x in (self.__m_sdfg, self.__m_sdfgHead, self.__m_sdfgInitState, self.__m_eqnTranslators, self.__m_jaxNameMap)]
        if(any(isAllocated)):
            assert all(isAllocated) or (not beStrict)
            return True
        return False
    # end def: _isStateAllocated



    ##########################################
    #   Variable Management
    #

    def _addArray(
            self,
            arg,
            isTransient: bool = True,
            altName: Optional[str] = None,
            forceArray: Optional[bool] = None,
            forceStorageType: None = None,
    ) -> str:
        """Creates an array inside Dace for `arg` and return its name.

        Note that this function, by defaults creates transients, which is different from `SDFG.add_array()`, that generates non transients by default.
        The function also distinguishes between `Scalar`s (having empty shapes) and `Array`s (non-empty shape)
        and calls `SDFG.add_scalar()` or `SDFG.add_array()` respectively.
        However, by setting `forceArray` to `True` the function will turn a `Scalar` into a one element `Array`.

        By default the name of the variable is `str(arg)`, but this not guaranteed.
        For example if that name would be a forbidden name, i.e. a `C++` keyword, the will try to find a new one.
        In case the variable name is given through `altName` a variable with that name will be created or an error is generated.
        In any case teh function will return the name that was finally used.

        Returns:
            The name of the array inside `self.__m_sdfg`.

        Args:
            arg:                The Jax object that should be maped to dace.
            isTransient:        If a transent should be created, by default.
            altName:            Try to create the variable with this name.
            forceArray:         Turn scalar in one element arrays.
            forceStorageType:   This parameter is ignored and if not `None` an error is issued.

        Notes:
            This function does not update the internal variable map, thus you should not use it, except you know what you are doing.
                Instead you should use `_createInitialInputs()`, `_createConstants()` or `_createJaxVarList()`, that updates the map.
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
        if(forceStorageType is not None):
            print(f"The `forceStorageType` is ignored remove it.", file=sys.stderr, flush=True)
        if((altName is not None) and (altName in self._forbiddenNames)):
            raise ValueError(f"You used `altName` to create the forbidden name `{altName}`.")
        #

        # This is the proposed name of the array, we will perform some checks
        #  and if needed rewritting furtherfurther down.
        argName = str(arg) if altName is None else str(altName)

        if(len(argName) == 0):
            raise ValueError(f"Got an empty name.")
        elif(not all([ x.isalnum() or x == '_'  for x in argName])):
            raise ValueError(f"The requested variable name `{argName}` contained invalid characters.")
        elif(argName[0].isdigit()):
            raise ValueError(f"Requested to create the array '{argName}', is ilegal since it starts with a digit.")
        elif(any([x.isspace()  for x in argName])):
            raise ValueError(f"The name of the array, '{argName}', to create contained a space!")
        #

        # Based on the provided name try to derive a replacmeent name.
        if(argName in self._forbiddenNames):
            nameTmpl = '_' + argName + '__{}'
            for iCounter in range(1000):
                _argName = nameTmpl.format(iCounter)
                if(_argName not in self._forbiddenNames):
                    argName = _argName
                    break
            else:
                raise ValueError(f"Failed to find a replacement name for '{argName}'")
            del iCounter, _argName
        #

        if argName in self.__m_sdfg.arrays:
            raise ValueError(f"The variable `{str(argName)}` is already recorded in the SDFG.")
        #

        shape     = arg.aval.shape                          # Shape of the array
        offset    = None                                    # i.e. no offset
        strides   = None                                    # TODO(phimuell): make it fully dynamic by using symbols and let dace figuring it out.
        storage   = dace.StorageType.Default                # Will be specialized later by the transformations.
        is_scalar = (shape == ())
        dtype     = self._translateDType(arg.aval.dtype)

        if(is_scalar and forceArray):       # "cast" the argument to an array.
            shape     = (1, )
            is_scalar = False
        #

        if(is_scalar):
            self.__m_sdfg.add_scalar(
                    name=argName,
                    storage=storage, dtype=dtype, transient=isTransient
            )
        else:
            self.__m_sdfg.add_array(
                    name=argName,
                    shape=shape, strides=strides, offset=offset,
                    storage=storage, dtype=dtype, transient=isTransient
            )
        #
        assert argName in self.__m_sdfg.arrays        # Final check
        return argName
    # end def: _addArray


    def _createInitialInputs(
            self, 
            jaxpr: ClosedJaxpr,
            sclar_as_array: bool = False,
            isTransient: bool = True,
    ):
        """Creates the initial inputs, i.e. arguments to the entier Jax expression.

        By default the variables created by this function are transients, which might be a bit surprising.

        There is no function to create the inputs for the indivisual equations,
        since their arguments will aready exists because:
        - they are either initial input
        - literals
        - former outputs of some equations.

        Notes:
            My setting `sclar_as_array` to `True` all sclar arguments will be turned into an array with shape `(1, )`.
                This is important for GPU arguments (somehow).
            This function updates the internal variable map.
        """
        from sys import stderr

        if(self.__m_jaxNameMap is None):          # Ensure that the name map is active.
            self.__m_jaxNameMap = dict()
        if(not isinstance(self.__m_sdfg, dace.SDFG)):
            raise TypeError(f"The internal SDFG object is not an SDFG but '{type(self.__m_sdfg)}'")
        if(len(self.__m_sdfg.arg_names) != 0):
            raise ValueError(f"Expected that the argument list of the SDFG is empty but it already contains: {self.__m_sdfg.arg_names}")
        #

        # Transfering scalars to the GPU seams a bit of a problem, so we ensure that arrays are created.
        #  The same is done for the output variable anyway.
        forceArray = sclar_as_array

        # We have to iterate through the non closed jaxpr, because there the names are removed.
        for inp in jaxpr.jaxpr.invars:
            name = self._addArray(inp, isTransient=isTransient, forceArray=forceArray)
            self.__m_sdfg.arg_names.append(name)
            self.__m_jaxNameMap[str(inp)] = name      # Add the name translation to the map.
        #
        return
    # end def: _createInitialInputs


    def _createConstants(
            self,
            jaxpr: ClosedJaxpr,
            device: dace.DeviceType = dace.DeviceType.CPU,
    ):
        """This function creates the constants, that are named in the closure.

        The function will create a constant in the `SDFG` and also create an array
        with name `__const_${cJaxName}`, where `cJaxName` is the jax name of the constant.
        In addition it will also update the variable map.
        """
        from copy import deepcopy
        assert self.__m_jaxNameMap is not None

        if(len(jaxpr.consts) == 0):
            return
        #
        if(device is dace.DeviceType.GPU):
            raise NotImplementedError("Constants are only implemented on CPU and not on GPU."
                                      " But it seems that DaCe can not handle them as well.")
        #

        # Interestingly the values and the names of the constants are kind of separated
        for cJaxVar, cValue in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            cJaxName = str(cJaxVar)         # Name of the variable in JAX

            # Now we create an array inside the SDFG, but with a special name.
            #  We add the two underscore to indicate that it is an internal.
            cDaCeName  = self._addArray(cJaxVar,
                                        isTransient=True,
                                        altName=f"__const_{cJaxName}",
            )

            # We have to pass the data descriptor to `add_constant` to link the array with the constant.
            #  If we would not do that, we would have both of them; this is something that is not even documented.
            self.__m_sdfg.add_constant(cDaCeName, deepcopy(cValue), self.__m_sdfg.arrays[cDaCeName])

            # And now we add it to the map.
            self.__m_jaxNameMap[cJaxName] = cDaCeName
        #
        return
    # end def: _createConstants


    def _createJaxVarList(
            self,
            jaxVarList,
            prevent_creation: bool = False,
            only_creation: bool = False,
    ) -> list[str]:
        """This function creates the listed jax variables and returns the SDFG names as a list.

        Expected input arguments are `JaxprEqn.invars` or `JaxprEqn.outvars`.
        The function will iterate through the list and return a `list` each element referes to the correspomnding SDFG variable.
        If a variable is a literal, then the corresponding entry in teh list will be `None`.

        If the JAX variable does not exists yet, it will be created and the variable map be updated.
        """
        assert self.__m_jaxNameMap is not None, "The variable map is not initialized."

        retList = []
        for var in jaxVarList:
            if isinstance(var, jax._src.core.Literal):              # Literal, there is no SDFG variable for this 
                if(only_creation):
                    raise ValueError(f"Requedsted `only_creation`, but `{str(var)}` is a literal and no creation is needed.")
                retList.append(None)
            elif isinstance(var, jax._src.core.Var):
                if(str(var) in self.__m_jaxNameMap):                # The variable is known, so we just return the SDFG name.
                    if(only_creation):
                        raise ValueError(f"Requested `only_creation`, but `{str(var)}` already exists as `{self.__m_jaxNameMap[str(var)]}`.")
                    retList.append( self.__m_jaxNameMap[str(var)] )
                else:
                    if(prevent_creation):
                        raise ValueError(f"Forbid the creation of variables, but need to create `{str(var)}`.")
                    retList.append( self._addArray(var) )           # The variable is not known, so we have to create it
                    self.__m_jaxNameMap[str(var)] = retList[-1]     #  and add it to the mapping.
            else:
                raise ValueError(f"The translation process is not implemented for '{type(var)}'")
            #
        # end for(var):
        return retList
    # end def: _createJaxVarList


    def _export_state(
            self,
            *, 
            jaxpr: Optional[ClosedJaxpr],
            doCleaning: bool = True
    ) -> 'TranslatedSDFG':
        """This function will generate a `TranslatedSDFG` form `self`.

        This function turns `self` into an `TranslatedSDFG` instance and then clears the internal state of `self`.
        However, by setting `doCleaning` to `False` no cleaning operation will be done.
        Note that in this case the state of `self` and the one in the returned `TranslatedSDFG` are shared.

        Notes:
            This function accounts for the renaming in case of empty `Jaxpr`.
        """
        if(not self._isStateAllocated()):
            raise ValueError(f"The state of `self` is not allocated, thus can not export it.")
        assert self.__m_sdfg is not None

        if(jaxpr is None):
            inpNames = None
            outNames = None

        elif(isinstance(jaxpr, ClosedJaxpr)):
            # Turn the Jax variable instances into a string.
            jaxOutNames = [str(out)  for out in jaxpr.jaxpr.outvars]
            jaxInpNames = [str(out)  for out in jaxpr.jaxpr.invars]

            # In the case of an empty `Jaxpr`, i.e. no equations, the Jax output variables are no
            #  longer given by `jaxpr.outvars` instead we have to rename them to the fake output variables
            #  that we have created in the `_handle_null_jaxpr()` function.
            if(len(jaxpr.eqns) == 0):
                jaxOutNames = [f'_zero_equation_hack_for_{out}'  for out in jaxOutNames]
            #

            # Now translate the Jax names to the SDFG names.
            inpNames = [self.__m_jaxNameMap[inp]  for inp in jaxInpNames]
            outNames = [self.__m_jaxNameMap[out]  for out in jaxOutNames]

        else:
            raise TypeError(f"Can not handle type `{type(jaxpr).__name__}` as `jaxpr` argument in the exporting.")
        #

        retVal = TranslatedSDFG(
            sdfg=self.__m_sdfg,
            startState=self.__m_sdfgInitState,
            finState=self.__m_sdfgHead,
            jaxNameMap=self.__m_jaxNameMap,
            inpNames=inpNames,
            outNames=outNames,
        )

        if(doCleaning):         # if requested clean the state of `self`.
            self._clearState()
            assert self.__m_sdfg is None
            assert retVal.sdfg is not None
        #
        return retVal
    # end def: _export_state


    def _load_state(self, translatedSDFG: TranslatedSDFG):
        """Allocates the internal state of `self` to the one given in `translatedSDFG`.

        This is a function that directly operates on `self`.
        There is also a static version, `_restore_from()`, which creates a new instance.

        Args:
            translatedSDFG:     The result of a previous call to `JaxprBaseTranslator._translateJaxpr()`.

        Notes:
            This is an internal function, you should only use it if you know what you are doing.
                It is recomended to use the `rebuildFrom()` function which constructs a new instance.
        """
        if(self._isStateAllocated()):
            raise ValueError(f"Can not load a state, `self` is still allocated.")
        #
        self._allocState()

        # Now we restore all the internal variables
        self.__m_sdfg = translatedSDFG.sdfg
        self.__m_sdfgInitState = translatedSDFG.startState 
        self.__m_sdfgHead = translatedSDFG.finState
        self.__m_jaxNameMap = translatedSDFG.jaxNameMap

        return self
    # end def: _load_state






    ####################################
    #   Internal Translation Routines
    #

    def _translateJaxprInternal(
            self,
            jaxpr: ClosedJaxpr,
            inp_sclar_as_array: bool,
            device: dace.DeviceType = dace.DeviceType.CPU,
    ):
        """This function does the actuall transformation.

        You should not use this function directly, instead you should always call `self._translate()`.
        The function will roughly do the following:
        - Creation of the SDFG.
        - Creation of all input arguments.
        - Creation of all constants.
        - It will go through the equations and using the underlining translator to turn them into an SDFG.

        It is important that the function will not return anything but store everything inside the state of `self`.

        Args:
            jaxpr:                  The `Jaxpr` that should be translated.
            inp_sclar_as_array:     Influences how sclar input arguments should be handled.
        """
        if(any([x is not None  for x in (self.__m_sdfg, self.__m_sdfgHead, self.__m_sdfgInitState)])):
            raise RuntimeError("Expected the that `self` is in an initial state, but it does not seem to be the case.")
        if(not self._isStateAllocated(beStrict=False)):
            raise RuntimeError("The internals are not allocated.")
        if(not isinstance(jaxpr, ClosedJaxpr)):
            raise TypeError(f"Expected a `jax.core.ClosedJaxp` instance but got `{type(jaxpr)}`")
        if(len(jaxpr.effects) != 0):
            raise ValueError(f"Currently `Jaxpr` instances with side effects are not supported.")
        if(len(jaxpr.out_avals) == 0):
            raise ValueError(f"You have zero output variables.")
        if(not jax.config.read("jax_enable_x64")):
            raise ValueError(f"The translation only works if `jax_enable_x64` is enabled. Do it manually or use `self.transform()`!")
        #

        # Creation of teh SDFG and the internal states.
        self.__m_sdfg = dace.SDFG(name=f"jax_{id(jaxpr)}")
        self.__m_sdfgInitState = self.__m_sdfg.add_state(label="initial_state", is_start_state=True)
        self.__m_sdfgHead = self.__m_sdfgInitState

        # Now we are creating the initial inputs, i.e. the ones that are named in the closure and are the arguments.
        #  Depending on the state turn them into shape one arrays.
        #  Important: They are created as transients.
        self._createInitialInputs(jaxpr, sclar_as_array=inp_sclar_as_array)

        # Now we are creating the constants.
        if len(jaxpr.consts) != 0:
            self._createConstants(jaxpr, device)
        #

        # This is a special corner case that might be occure in some lambdas.
        if(len(jaxpr.jaxpr.eqns) == 0):
            self._handle_null_jaxpr(jaxpr)
            return
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

        return
    # end def: _translateJaxprInternal


    def _translateEqn(
            self,
            closedJaxp: ClosedJaxpr,
            eqn: JaxprEqn,
    ):
        """This function translates the equation (statement) to an SDFG state.

        The new state will be added directly after the current head.
        It is important that the non closed equation is passed to this function, while the closed jaxpr is passed.
        This function will also modify the current head.
        """
        assert isinstance(eqn, jax._src.core.JaxprEqn)
        assert len(eqn.outvars) >= 1, f"Expected to find at least one output variable for equation '{str(eqn)}' but it had {len(eqn.outvars)}"
        assert all([str(out) not in self.__m_jaxNameMap  for out in eqn.outvars]), f"The outputs {[str(out)  for out in eqn.outvars if str(out) in self.__m_jaxNameMap]} were already created."
        assert len(eqn.effects) == 0, "This class can only handle siode efect free equations."
        assert self.__m_sdfgHead is not None, "The head of the SDFG state machine is `None`."
        assert self.__m_sdfg     is not None, "The SDFG is not allocated."

        # Inside this state we will add everything that is related to this equation.
        eqnState = self._appendNewState(label=f'{eqn.primitive.name}_{str(eqn.outvars[0])}__{id(eqn)}')

        # We now create the name list for the variables
        inVarNames  = self._createJaxVarList(eqn.invars,  prevent_creation=True)
        outVarNames = self._createJaxVarList(eqn.outvars, only_creation=True)
        assert all([(o is not None) and (o in self.__m_sdfg.arrays)  for o in outVarNames])

        # Now we look for the translator that can handle the primitive
        for eqnTranslator in self.__m_eqnTranslators:
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
        self.__m_sdfgHead = newSDFGHead

        return
    # end def: _translateEqn


    def _handle_null_jaxpr(
            self,
            jaxpr: ClosedJaxpr,
    ):
        """This function is called in case a `Jaxpr` with zero equations is encountered.

        In such a `Jaxpr` an input is just forwarded as output, however, your implementation and dace can not handle this.
        The solution is that the input is first copied into a temprary which is then considered as true Jax output.
        The bad thing about this is, that the names denoted in `jaxpr.jaxpr.outvars` can no longer be used to get the output names.
        Even worse, the names used in `jaxpr.jaxpr.outvars` are known to the internal Jax name to SDFG name, but they are not the outputs.
        However, since this is a very edgy corner case this should not matter, in addition names of Jax variables are not used after the translation.

        To get the true names of teh output you can use the names that are denoted in the `outNames` of the `TranslatedSDFG` object that is generated.
        """
        if(len(jaxpr.jaxpr.eqns) != 0):
            raise RuntimeError(f"What the hell are you doing, calling the `_handle_null_jaxpr()` on a `jaxpr` with {len(jaxpr.eqns)} equations!")
        if(len(jaxpr.out_avals) == 0):
            raise ValueError(f"Seriously, you have a `Jaxpr` with zero outputs.")
        #
        print("WARNING: Detected an `Jaxpr` with no equations, will now perform a nasty renaming of the output variables.\n"
              "          You should consult `JaxprBaseTranslator._handle_null_jaxpr()` in case of errors.",
              flush=True, file=sys.stderr
        )

        for jaxOutVar in jaxpr.jaxpr.outvars:
            # Create the fake Jax output variable and create an sdfg variable for it.
            altJaxVarName = f'_zero_equation_hack_for_{str(jaxOutVar)}'
            orgJaxVarSDFGName = self.__m_jaxNameMap[str(jaxOutVar)]
            _ = self._addArray(jaxOutVar, isTransient=True, altName=altJaxVarName)
            self.__m_jaxNameMap[altJaxVarName] = altJaxVarName

            # Now copy the input into the fake output variable.
            inpAcc = self.__m_sdfgHead.add_read(orgJaxVarSDFGName)
            outAcc = self.__m_sdfgHead.add_write(self.__m_jaxNameMap[altJaxVarName])
            self.__m_sdfgHead.add_nedge(
                    src=inpAcc,
                    dst=outAcc,
                    data=dace.Memlet.from_array(orgJaxVarSDFGName, self.__m_sdfg.arrays[orgJaxVarSDFGName])
            )
        # end for:

        return
    # end def: _handle_null_jaxpr




    ##################################
    #   Getter
    #

    def getSDFG(self):
        """Returns the SDFG of self.
        """
        assert self.__m_sdfg is not None, "The SDFG object is `None` are you sure that the translation is active."
        return self.__m_sdfg
    # end def: getSDFG


    def getArray(self, name):
        """Return the array `name` inside `self`.

        Effectively a shorthand of `self.getSDFG().arrays[name]`.
        """
        assert self.__m_sdfg is not None, "The SDFG object is `None` are you sure that the translation is active."
        return self.__m_sdfg.arrays[name]
    # end def: getArray




    ##################################
    #   Misc
    #






    def _appendNewState(self, label: str) -> dace.SDFGState:
        """This function creates a new after the current head with name `name`.
        """
        assert self.__m_sdfg is not None, "The SDFG object is `None` are you sure that the translation is active."
        assert self.__m_sdfgHead is not None, "The SDFG head state is `None`."
        assert isinstance(label, str) and label
        
        new_state = self.__m_sdfg.add_state_after(self.__m_sdfgHead, label=label)
        return new_state
    # end def: _appendNewState


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


    ####################################
    #   Forbidden variable names

    _forbiddenNames: set[str] = {
        # These should be most of the C++ keywords, it is more important to have the short ones.
        #  Taken from `https://learn.microsoft.com/en-us/cpp/cpp/keywords-cpp?view=msvc-170`
        'alignas', 'alignof', 'and', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch',
        'char', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 'continue',
        'decltype', 'default', 'delete', 'directive', 'do', 'double', 'else', 'enum', 'explicit', 'export',
        'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'noexcept', 'not', 'nullptr', 'operator', 'or', 'private', 'protected',
        'public', 'register', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 'struct',
        'switch', 'template', 'this', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union',
        'unsigned', 'using', 'using', 'virtual', 'void', 'volatile', 'while', 'xor', 'std',
    }
# end class(JaxprToSDFG):
   
