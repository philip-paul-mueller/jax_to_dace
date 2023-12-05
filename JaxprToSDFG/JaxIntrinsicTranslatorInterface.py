"""This files contains the interface for translating Jax Intrinsics to SDFG parts.
"""

from jax._src.core import JaxprEqn
from typing import Union
import dace


class JaxIntrinsicTranslatorInterface:
    """This class is serves as the foundation of all intrinsic traslators.

    An intrinsic translator basically handles the translation of a single equation to its equivalent SDFG construct.
    There might be many different translators depending on what equation is currently processed.

    An instance must have the following properties:
    - It must be stateless.
    - It must be possible to construct it without any arguments.
    - It must be derived from `JaxIntrinsicTranslatorInterface`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a structure.

        It is required that subclasses calls this method during initialization.
        """
        pass
    # end def: __init__


    def canHandle(self,
                  translator,
                  inVarNames: list[Union[str, None]],
                  outVarNames: list[str],
                  eqn: JaxprEqn,
    ):
        """Tests if `self` is able to translate `eqn` into a corresponding SDFG construct.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.

        Notes:
            This function should consider `self` and all of its arguments as constant.
        """
        raise NotImplementedError("You have to implement this function.")
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """This function is used to actually translate `eqn` into an SDFG.

        This function is called after the translator has created named variables, i.e. arrays, for the inputs and the outputs.
        The names are stored inside the `inVarNames` and `outVarNames`, respectively, which may be `None`.

        Returns:
            The function returns the last equation state, i.e. the value that will be passed as `eqnState` to the next translation.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.

        Notes:
            The `{in,out}VarNames` have the same order as `eqn.{in,out}vars`.
            It is possible that entries in `inVarNames` are `None` which means that the argument is a literal and no array inside the SDFG was created for it.
                It is possible that `inVarNames` may only contains `None`, which is an edge case, but must be handled.
                `outVarNames` will never contain `None` values.
            Returning the `eqnState` allows the function to modify, i.e. add more states, as just one state.
                In previous versions this was not requiered, as a compability feature if `None` is returned it is assumed that `eqnState` is returned.
        """
        raise NotImplementedError("You have to implement this function.")
    # end def: translateEqn

# end class(JaxIntrinsicTranslatorInterface):


