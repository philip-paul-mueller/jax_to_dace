import numpy as np
import jax
import dace
import sys

from typing import Optional

from dace.dtypes    import DeviceType
from jax._src.core  import ClosedJaxpr, JaxprEqn, Jaxpr

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from JaxprToSDFG._JaxprBaseTranslator               import JaxprBaseTranslator
from JaxprToSDFG._translatedSDFG                    import TranslatedSDFG


class JaxprToSDFG(JaxprBaseTranslator):
    """This is a simple class that allows to translate an `jaxpr` instance to an SDFG.

    This class, which is an extension to the `JaxprBaseTranslator` should be seen as a driver code.
    It extends the `JaxprBaseTranslator` with the follwoing:
    - Crates return statements.
    - Perform optimizations.
    - Turn the SDFG into a GPU one, by default it is for CPU.

    Notes:
        If a translation failes the internal state is not deallocated, before you can use the object again,
            you have to call `_clearState()` manually.

    Todos:
        Implement a JIT semantic.
        Turn `JaxprBaseTranslator` into a member not a base class.
    """


    ########################
    #       Initialization
    #

    def __init__(
            self,
            device: DeviceType = DeviceType.CPU,
            inp_on_gpu: bool = False,
    ):
        """`self` is generally stateless, but maintains some variables during translation.

        In addition there are some default values, that are permanently stored inside `self` and set through the constructor.
        The transformsers are set up in `_initEqnTranslators()`.

        Args:
            device:     The default device for which we should generate, defaults to `CPU`.
            inp_on_gpu  The default value for the `inp_on_gpu` argument of the `transform()` function.
        """
        super().__init__()

        self.m_def_device     = device
        self.m_def_inp_on_gpu = inp_on_gpu
    # end def: __init__



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
                  inp_on_gpu: Optional[bool] = None,
    ) -> dace.SDFG:
        """Transforms `jaxpr` into an `SDFG`.

        By default the function just performs some verbatim transformation.
        By setting `simplify` to `True` the function will call `simplify()` on the generated `SDFG`.
        Furthemore, by setting `auto_opt` to `True` the function will call `auto_optimize` on the `SDFG`.
        If `auto_opt` is an integer, that many times `auto_optimize` will be applied.

        This function applies several DaCe transformations, some of them accepts a `validate` keyword.
        You can influence the value that is passed to it through the `iValidate` argument,
        however, in any case the function will perform a final validation.
        By default this value is set to `None` in which case the behaviour depends on the device.
        If the device is CPU then it is equal to `True` if it is GPU then it is equal to `False`.

        Even in case `device` was set to `DeviceType.GPU` the generated SDFG assumes that the input arguments are on the host.
        Thus compiling it will insert code that will first copy them from the CPU to the GPU and then copy the result value back.
        However, by setting `inp_on_gpu` to `True` the input arguments will be already on GPU, it is the responsibility of the user to ensure that this is the case.
        If that argument is `None`, the default, then the value of `inp_on_gpu` passed during construction will be used.

        Args:
            jaxpr:          The `ClosedJaxpr` instance that should be translated.
            simplify:       Apply simplify to the generated `SDFG`.
            auto_opt:       Appy `auto_optimize` on the `SDFG` before returning it.
            device:         For which device to optimize for, if `None` the default one is used.
            iValidate:      Controles if _intermediate_ validation should be performed.
            ret_by_arg:     Return the result by arguments, defaults to `False`.
            inp_on_gpu:     In GPU mode the inputs _and_ output arguments are expected to be on the GPU, ignored otherwhise.

        Notes:
            If `ret_by_arg` is set to `True` then the SDFG will transform the return statement `return A`
                into an assignement, `_out[:] = A[:]`, where `_out` is a pseudo argument that is added at
                the end to the argument list. If multiple values are returned then the variables will be
                named `_out{i}`.
            The strange behaviour of the `iValidate=None` is due to some strange behaviour in DaCe.
                Furthermore this whole solution is not a permanent one.
                However, a normal user should probably never use this argument.
        """
        import dace
        from dace import SDFG
        from dace.transformation.auto.auto_optimize import auto_optimize

        if(not jax.config.jax_enable_x64):
            raise ValueError("`x64` Support was disabled, you have to enable it by calling `jax.config.update('jax_enable_x64', True)` before anything else.")
        if(not isinstance(jaxpr, ClosedJaxpr)):
            raise TypeError(f"The `jaxpr` you passed was not a `ClosedJaxpr` instance, you have to applied `jax.make_jaxpr()` to it first and concretize it.")
        #

        if(inp_on_gpu is None):
            inp_on_gpu = self.m_def_inp_on_gpu
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

        # Controles if we perform intermediate validation.
        if(iValidate is None):
            if(device == DeviceType.GPU):   intermediate_validation = False
            else:                           intermediate_validation = True
        else:
            intermediate_validation = iValidate
        #

        # Now we generate the SDFG and put it into an `TranslatedSDFG` object.
        #  However, we will restore it afterwards again, which is a bit stupid.
        #  But I do not like the idea of adding a flag to `_translateJaxpr()` to
        #  prevent dealocation, I want that this feature is explicitly.
        translatedSDFG: TranslatedSDFG = self.translateJaxpr(
                jaxpr=jaxpr,
                sclar_as_array=bool(device == DeviceType.GPU),
                device=device,
        )

        # Now we restore the internal of the translator.
        #  This is esentially needed because we want to create the return values.
        self._load_state(translatedSDFG)              ; del translatedSDFG

        # Now we are creating the return arguments.
        self._createReturnOutput(jaxpr, ret_by_arg=ret_by_arg)

        # This is the SDFG we have created.
        jaxSDFG = self.getSDFG()

        if(simplify):
            jaxSDFG.simplify(validate=intermediate_validation)
        for _ in range(auto_opt):
            # Regardless if we are on GPU or not, we always optimize for CPU.
            #  It is basically the same idea as we used for `intermediate_validation==False` in that case.
            #  The deeper reason is, if we would set it to `GPU` then we get errors in some validation process.
            auto_optimize(sdfg=jaxSDFG, device=dace.DeviceType.CPU, validate=intermediate_validation)
        #
        if(device is dace.DeviceType.GPU):   # If needed we will now apply some simplifications to teh SDFG to make it GPU ready
            jaxSDFG.apply_gpu_transformations(validate=intermediate_validation, validate_all=False)
            if(inp_on_gpu):
                self._relocateSignatureArgsToGPU(jaxpr=jaxpr, validate=intermediate_validation)
            jaxSDFG.simplify(validate=intermediate_validation, validate_all=False)  # The documentation recommends this.
        #

        # Since we have disabled all validations until now we now have to ensure that everything went well.
        jaxSDFG.validate()

        # Now we can clean `self`.
        #  We do not need anything else from our base class.
        self._clearState()

        return jaxSDFG
    # end def: transform


    def __call__(self, jaxpr: ClosedJaxpr, *args, **kwargs) -> dace.SDFG:
        """An alias for `self.transform(jaxpr, *args, **kwargs)`.
        """
        return self.transform(jaxpr, *args, **kwargs)
    # end def: __call__



    def _createReturnOutput(self,
                            jaxpr: ClosedJaxpr,
                            ret_by_arg: bool
    ):
        """Creates the return value statement.

        This function will always allocate the return variables on the host.
        For relocating them to the GPU use `_relocateSignatureArgsToGPU()`.

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
        final_state = self._appendNewState(label='Return_State')

        for jVar, sVar in outVarMap.items():
            jAN    = final_state.add_read(jVar)
            sAN    = final_state.add_write(sVar)
            memlet = dace.Memlet.from_array(sVar, self.getArray(jVar))
            final_state.add_edge(jAN, None, sAN, None, memlet)                      # Now we add  the connection between them
        #

        # If needed add the pseudo arguments to the argument list
        if(ret_by_arg):
            sdfd = self.getSDFG()
            assert len(sdfg.arg_names) > 0
            for sVar in sdfgOutVarOrder:
                sdfg.arg_names.append(sVar)
        #

        return
    # end def: _createReturnOutput


    def _relocateSignatureArgsToGPU(
            self,
            jaxpr: ClosedJaxpr,
            validate: bool,
    ):
        """This function "relocates" the input and output arguments from the host to the CPU.

        Basically this is done by interating through the list of arrays and change their storage.
        Afterwards the function performs a simplification step.

        Notes:
            This function _must_ be run after the GPU transformations were applied.
        """
        sdfg = self.getSDFG()

        # Set all arguments to global storage.
        for sdfgName in sdfg.arg_names:
            assert sdfgName in sdfg.arrays
            sdfgArray = self.getArray(sdfgName)
            sdfgArray.storage = dace.StorageType.GPU_Global
        # end for(sdfgName):

        # For the automatic copying of variables DaCe created transients on the GPU, (prfixed with `gpu_`).
        #  They are now unnecessary and we get rid of them by calling the simplificiation step.
        sdfg.simplify(validate_all=validate)

        return
    # end def: _relocateSignatureArgsToGPU


# end class(JaxprToSDFG):
   
