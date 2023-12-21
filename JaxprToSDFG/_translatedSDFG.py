"""Contains the `TranslatedSDFG` class which is the result of a translation.
"""
import dace
from dataclasses import dataclass

from typing import Optional


@dataclass(init=True, repr=True, eq=False, frozen=False, kw_only=True, slots=True)
class TranslatedSDFG:
    """This class is used as return argument of the translation.

    The members have the following meaning:
    - `sdfg` the `SDFG` object that was created.
    - `statrtState` the first state in the `SDFG` state machine.
    - `finState` the last state in the state machine.
    - `jaxNameMap` a `dict` that maps every JAX name to its corresponding SDFG variable name.
    - `inpNames` a `list` of the `SDFG` variables that are used as input.
    - `outNames` a `list` of the `SDFG` variables that are used as output.
    """
    sdfg:           dace.SDFG
    startState:     dace.SDFGState
    finState:       dace.SDFGState
    jaxNameMap:     dict[str, str]
    inpNames:       Optional[list[str]]
    outNames:       Optional[list[str]]

    def __getitem__(self, idx: str):
        if(not isinstance(idx, str)):
            raise TypeError(f"Expected `idx` as `str` but got `{type(str)}`")
        if(not hasattr(self, idx)):
            raise KeyError(f"The key `{idx}` is not known.")
        return getattr(self, idx)
    # end def: __getitem__


    def validate(self):
        """Validate the underlying SDFG.

        Before this function calls `validate()` on the SDFG, it will turn any input array into a non transient one.
        This is done to prevent the issuing of 'non initialized' data warnings in the validation.
        """
        orgTransState: dict[str, bool] = {}
        for var in self.inpNames:
            orgTransState[var] = self.sdfg.arrays[var].transient
            self.sdfg.arrays[var].transient = False
        #

        try:
            self.sdfg.validate()
        finally:
            for var, orgValue in orgTransState.items():
                self.sdfg.arrays[var].transient = orgValue
        #

        return True
    # end def: validate

# end class(TranslatedSDFG):


