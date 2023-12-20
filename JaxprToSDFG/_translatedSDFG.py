"""Contains the `TranslatedSDFG` class which is the result of a translation.
"""
import dace
from dataclasses import dataclass

from typing import Optional


@dataclass(init=True, repr=True, eq=False, frozen=False, kw_only=True, slots=True)
class TranslatedSDFG:
    """This class is used as return argument of the translation.
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

# end class(TranslatedSDFG):


