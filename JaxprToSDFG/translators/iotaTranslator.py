"""This file contains the iota command.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union

from sys import stderr


class IotaTranslator(JaxIntrinsicTranslatorInterface):
    """Implements the iota.

    It basically generates the sequence `0, 1, ...` along a given axis.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a cast translator.
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
        return str(eqn.primitive.name) == "iota"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Creates the sequence.

        The dimension along which the sequence will be generated is passed by the parameter `dimension`.
        The other two parametes (`dtype` and `shape`) are ignored.

        For this a single tasklet with an assignement from a map loop symbol will be generated.

        Args:
            translator:     The `JaxprToSDFG` instance that is respnsible for the translation.
            inVarNames:     List of the names of the arrays created inside the SDFG for the inpts.
            outVarNames:    List of the names of the arrays created inside the SDFG for the outputs.
            eqn:            The `JaxprEqn` instance that is currently being handled.
            eqnState:       This is the SDFG State into which the translation should happen.
        """
        iotaDim   = eqn.params['dimension']         # along which dimension the increase happens.
        outName   = outVarNames[0]
        outShape  = eqn.outvars[0].aval.shape
        iotaItVar = '__iota'                        # This is the index of the loop that will be used for assigning.


        if(len(eqn.invars) != 0):
            raise ValueError(f"`iota` does not take any arguments.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(len(outShape) == 0):
            raise ValueError(f"In translating iota for `{outName}` the shape had length zero.")
        if(len(outShape) != 1):
            print(f"Translation in many dimensions is not yet tested.")
        #


        # This is always the code that has to be generated
        tCode = f'__out = {iotaItVar}'

        _tOutputs   = []
        _tMapRanges = []
        for dim, dimSize in enumerate(outShape):
            # The iota dimension will always come first, and it can not be swapped.
            #  TODO: add a construct to tell DaCe that it can not swap the iteration order here.
            if(dim == iotaDim):
                whereToInsert = 0
                dimItVar = iotaItVar
            else:
                whereToInsert = -1
                dimItVar = f'__i{dim}'
            #

            _tMapRanges.insert(whereToInsert, (dimItVar, f'0:{dimSize}') )  # Only in the map ranges we must do that.
            _tOutputs.append(dimItVar)
        # end for:

        eqnState.add_mapped_tasklet(
            f'_iota_{outName}_',
            map_ranges=_tMapRanges,
            inputs={},  # No inputs are needed.
            code=tCode,
            outputs={'__out': dace.Memlet.simple(outName, ",".join(_tOutputs))},
            external_edges=True
        )
        return eqnState
    # end def: translateEqn

# end class(IotaTranslator):

