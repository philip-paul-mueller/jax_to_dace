import numpy as np
import jax
import dace
import sys

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
        If you start reading the code start at the `transform()` function.
        The class only has a (allocated) members during the transformatiosn, otherwhise they are `None`.
        If a translation failes the internal state is not deallocated, before you can use the object again,
            you have to call `_clearState()` manually.

    Todo:
        Fully dynamic storage sizes or just the strides(?), i.e. make them symbols such that DaCe can play more.
        Implement a JIT semantic.
    """


    ########################
    #       Initialization
    #

    def __init__(
            self,
            device: DeviceType = DeviceType.CPU
    ):
        """`self` is generally stateless, but maintains some variables during translation.

        In addition there are some default values, that are permanently stored inside `self` and set through the constructor.
        The transformsers are set up in `_initEqnTranslators()`.

        Args:
            device:     The default device for which we should generate, defaults to `CPU`.
        """
        # We now allocate the variables of the internal state (by calling the clear function)
        self._clearState()

        self.m_def_device = device
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

        """Device that we traget.
        """
        self.m_device = None


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

        # Add self to kwargs, such that classes can access it.
        #kwargs['_driver'] = self

        for cls in ALL_TRAFOS:
            self.m_eqnTranslators.append( cls(*args, **kwargs) )
        #
        return self
    # end def: _initEqnTranslators






    #################################################
    #   Translation interface
    #

    def transform(self,
                  jaxpr,
                  simplify: bool = False,
                  auto_opt: bool = False,
                  device: Optional[DeviceType] = None,
                  ret_by_arg: bool = False,
                  iValidate: Optional[bool] = None,
    ) -> dace.SDFG:
        """Transforms `jaxpr` into an `SDFG`.

        By default the function just performs some verbatim style transformation.
        By setting `simplify` to `True` the function will call `simplify()` on the generated `SDFG`.
        Furthemore, by setting `auto_opt` to `True` the function will call `auto_optimize` on the `SDFG`.
        If `auto_opt` is an integer, that many times `auto_optimize` will be applied.

        This function applies several DaCe transformations, some of them accepts a `validate` keyword.
        You can influence the value that is passed to it through the `iValidate` argument,
        however, in any case the function will perform a final validation.
        By default this value is set to `None` in which case the behaviour depends on the device.
        If the device is CPU then it is equal to `True` if it is GPU then it is equal to `False`.


        Args:
            jaxpr:          The `ClosedJaxpr` instance that should be translated.
            simplify:       Apply simplify to the generated `SDFG`.
            auto_opt:       Appy `auto_optimize` on the `SDFG` before returning it.
            device:         For which device to optimize for, if `None` the default one is used.
            iValidate:      Controles if _intermediate_ validation should be performed.
            ret_by_arg:     Return the result by arguments, defaults to `False`.

        Notes:
            If `ret_by_arg` is set to `True` then the SDFG will transform the return statement `return A`
                into an assignement, `_out[:] = A[:]`, where `_out` is a pseudo argument that is added at
                the end to the argument list. If multiple values are returned then the variables will be
                named `_out{i}`.
            The strange behaviour of the `iValidate=None` is due to some strange behaviour in DaCe.
                Furthermore this whole solution is not a permanent one.
        """
        import dace
        from dace import SDFG
        from dace.transformation.auto.auto_optimize import auto_optimize

        if(not jax.config.jax_enable_x64):
            raise ValueError("`x64` Support was disabled, you have to enable it by calling `jax.config.update('jax_enable_x64', True)` before anything else.")
        #

        if(not isinstance(jaxpr, ClosedJaxpr)):
            raise TypeError(f"The `jaxpr` you passed was not a `ClosedJaxpr` instance, you have to applied `jax.make_jaxpr()` to it first and concretize it.")
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
        assert getattr(self, "m_sdfg", None) is None

        try:
            self.m_device = self.m_def_device if device is None else device

            # Controles if we perform intermediate validation.
            if(iValidate is None):
                if(self.m_device == DeviceType.GPU):    intermediate_validation = False
                else:                                   intermediate_validation = True
            else:
                intermediate_validation = iValidate
            #

            jaxSDFG: SDFG = self._transform(jaxpr=jaxpr, ret_by_arg=ret_by_arg)   # Perform the translation.

            if(simplify):
                jaxSDFG.simplify(validate=intermediate_validation)
            for _ in range(auto_opt):
                # We make no validation, here to avoid some issue.
                jaxSDFG = auto_optimize(sdfg=jaxSDFG, device=self.m_device, validate=intermediate_validation)
            #
            if(self.m_device is dace.DeviceType.GPU):   # If needed we will now apply some simplifications to teh SDFG to make it GPU ready
                jaxSDFG.apply_gpu_transformations(validate=intermediate_validation, validate_all=False)
                jaxSDFG.simplify(validate=intermediate_validation, validate_all=False)
            #
            jaxSDFG.validate()      # This function throws if an error is detected.

        except:
            raise

        else:
            self._clearState()      # Not in `finally` to ensure that the state can be inspected after an error.
                                    #  TODO: Put it in the beginning as well, to avoid annoying error.

        return jaxSDFG
    # end def: transform


    def __call__(self, jaxpr: ClosedJaxpr, *args, **kwargs) -> dace.SDFG:
        """An alias for `self.transform(jaxpr, *args, **kwargs)`.
        """
        return self.transform(jaxpr, *args, **kwargs)
    # end def: __call__




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
            The name of the array inside `self.m_sdfg`.

        Args:
            arg:                The Jax object that should be maped to dace.
            isTransient:        If a transent should be created, by default.
            altName:            Try to create the variable with this name.
            forceArray:         Turn scalar in one element arrays.
            forceStorageType:   This parameter is ignored and if not `None` an error is issued.

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

        if argName in self.m_sdfg.arrays:
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
            self.m_sdfg.add_scalar(
                    name=argName,
                    storage=storage, dtype=dtype, transient=isTransient
            )
        else:
            self.m_sdfg.add_array(
                    name=argName,
                    shape=shape, strides=strides, offset=offset,
                    storage=storage, dtype=dtype, transient=isTransient
            )
        #
        assert argName in self.m_sdfg.arrays        # Final check
        return argName
    # end def: _addArray


    def _createInitialInputs(self, jaxpr: ClosedJaxpr):
        """Creates the initial inputs, i.e. arguments to the jax expression.

        The function will update the internal variable map.

        There is no function to create the inputs for the indivisual equations, since:
        - they are either initial input
        - literals
        - former outputs of some equations.
        """
        from sys import stderr

        if(self.m_jaxNameMap is None):          # Ensure that the name map is active.
            self.m_jaxNameMap = dict()
        if(not isinstance(self.m_sdfg, dace.SDFG)):
            raise TypeError(f"The internal SDFG object is not an SDFG but '{type(self.m_sdfg)}'")
        if(len(self.m_sdfg.arg_names) != 0):
            raise ValueError(f"Expected that the argument list of the SDFG is empty but it already contains: {self.m_sdfg.arg_names}")
        #

        # Transfering scalars to the GPU seams a bit of a problem, so we ensure that arrays are created.
        #  The same is done for the output variable anyway.
        forceArray = False
        if((self.m_device is dace.DeviceType.GPU) and any([len(inp.aval.shape) == 0  for inp in jaxpr.jaxpr.invars])):
            print(f"In GPU mode all scalar input variables are transformed into arrays with one element.", file=stderr, flush=True)
            forceArray = True
        #

        # We have to iterate through the non closed jaxpr, because there the names are removed.
        for inp in jaxpr.jaxpr.invars:
            name = self._addArray(inp, isTransient=False, forceArray=forceArray)
            self.m_sdfg.arg_names.append(name)
            self.m_jaxNameMap[str(inp)] = name      # Add the name translation to the map.
        #
        return
    # end def: _createInitialInputs


    def _createReturnOutput(self,
                            jaxpr: ClosedJaxpr,
                            ret_by_arg: bool
    ):
        """Creates the return value statement.

        Args:
            jaxpr:          The `ClosedJaxpr` for which the return statements should be created.
            ret_by_arg:     Create a pseudoargument to return the value instead, see `self.transform()` for more.
        """
        nbOutVars: int = len(jaxpr.jaxpr.outvars)
        retTuple: bool = nbOutVars > 1
        if(nbOutVars == 0):
            raise ValueError(f"Passed zero putput variables.")
        #

        # Determine the name of the return value.
        if(ret_by_arg):
            if(retTuple):   retValNameTempl: str = '_out{}'
            else:           retValNameTempl: str = '_out'            # Pseudoargument in which the value is returned.
        else:
            if(retTuple):   retValNameTempl: str = '__return_{}'
            else:           retValNameTempl: str = '__return'        # Special SDFG name.
        #

        # Create now the arrays that we use as output, these are the special `__return` / `__return_{IDX}` variables.
        outVarMap: dict[str, str] = {}
        sdfgOutVarOrder: list[str] = []
        for i in range(nbOutVars):
            jaxOutVar  = jaxpr.jaxpr.outvars[i]         # Name of the variable inside jax/SDFG
            SDFGoutVar = retValNameTempl.format(i)      # This name will mark it as a return value.

            # Create an output array that has the same shape as `jaxOutVar` but with name `SDFGoutVar`.
            #  We have to force the creation of a container (otherwhise the code generator will safe the result in a pass by value argument).
            _ = self._addArray(jaxOutVar, isTransient=False, altName=SDFGoutVar, forceArray=True)
            outVarMap[str(jaxOutVar)] = SDFGoutVar
            sdfgOutVarOrder.append(SDFGoutVar)
        # end for(i):

        # Now we create the return state.
        final_state = self.m_sdfg.add_state_after(self.m_sdfgHead, label='Final_State')

        for jVar, sVar in outVarMap.items():
            jAN    = final_state.add_read(jVar)
            sAN    = final_state.add_write(sVar)
            memlet = dace.Memlet.from_array(sVar, self.getArray(jVar))
            final_state.add_edge(jAN, None, sAN, None, memlet)                      # Now we add  the connection between them
        #

        # If needed add the pseudo arguments to the argument list
        if(ret_by_arg):
            assert len(self.m_sdfg.arg_names) > 0
            for sVar in sdfgOutVarOrder:
                self.m_sdfg.arg_names.append(sVar)
        #

        return
    # end def: _createReturnOutput


    def _createConstants(self, jaxpr: ClosedJaxpr):
        """This function creates the constants, that are named in the closure.
        """
        from copy import deepcopy
        assert self.m_jaxNameMap is not None

        if(len(jaxpr.consts) == 0):
            return
        #
        if(self.m_device is dace.DeviceType.GPU):
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
            self.m_sdfg.add_constant(cDaCeName, deepcopy(cValue), self.m_sdfg.arrays[cDaCeName])

            # And now we add it to the map.
            self.m_jaxNameMap[cJaxName] = cDaCeName
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

    def _transform(self,
                   jaxpr: ClosedJaxpr,
                   ret_by_arg: bool,
        ) -> dace.SDFG:
        """This function does the actuall transformation.

        You should not use this function directly, instead you should always call `self.transform()`.
        The reason is that `self.transform()` prepares the internal state of `self`.
        Also look there for more information about the arguments.
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
        self._createReturnOutput(jaxpr, ret_by_arg=ret_by_arg)

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
   
