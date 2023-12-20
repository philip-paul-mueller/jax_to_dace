"""Common stuff that is needed everywhere.
"""
import numpy as np
import dace
import jax

from typing import Optional, Any

from dace                           import DeviceType, SDFG, SDFGState
from dace.sdfg.nodes                import NestedSDFG
from jax._src.core                  import ClosedJaxpr, JaxprEqn, Jaxpr
from JaxprToSDFG._translatedSDFG    import TranslatedSDFG

def getNestedSDFGSymbolMapping(
        parentSDFG: SDFG,
        nestedState: SDFGState,
        translatedNestedSDFG: TranslatedSDFG,
) -> Optional[dict[str, Any]]:
    """Computes the symbol map for the SDFG.

    The map describes which internal symbol corresbonds to which outer symbol.

    Args:
        parentSDFG:                 The SDFG that will contain the SDFG.
        nestedState:                The state into which we will add the nested SDFG.
        translatedNestedSDFG:       The result of the translation of the lambda expression.

    Notes:
        Currently thsi returns `None`.
    """
    return None
# end def: getNestedSDFGSymbolMapping


def AddNestedSDFG(
        parentSDFG: SDFG,
        nestedState: SDFGState,
        translatedNestedSDFG: TranslatedSDFG,
        inputNameMapping: dict[str, str],
        outputNameMapping: dict[str, str],
        name: str,
) -> NestedSDFG:
    """This function adds a nested SDFG to state `nestedState`.

    Both the `inputNameMapping` and `outputNameMapping` mapps the name of an array
    outside to the nested SDFG to the name it should have inside the nested SDFG.
    This ordering makes sense for inputs but may be a bit confusing for the outputs.

    The function returns the nested SDFG object.

    Args:
        parentSDFG:                 The SDFG that will contain the SDFG.
        nestedState:                The state into which we will add the nested SDFG.
        translatedNestedSDFG:       The result of the translation of the lambda expression.
        inputNameMapping:           Name mapping for the inputs.
        outputNameMapping:          Name mapping for the outputs.
    """
    from dace.sdfg.propagation          import propagate_memlets_sdfg

    for nameOutside, nameInside in inputNameMapping.items():
        if(nameOutside not in parentSDFG.arrays):
            raise ValueError(f"Expected that `{nameOutside}` is inside the parent SDFG but it was not there (INSIDE) || {inputNameMapping}.")
        if(nameInside not in translatedNestedSDFG.sdfg.arrays):
            raise ValueError(f"Expected that `{nameInside}` is inside the nested SDFG but it was not there (INSIDE) || {inputNameMapping}.")
    for nameOutside, nameInside in outputNameMapping.items():
        if(nameOutside not in parentSDFG.arrays):
            raise ValueError(f"Expected that `{nameOutside}` is inside the parent SDFG but it was not there (OUTSIDE) || {outputNameMapping}.")
        if(nameInside not in translatedNestedSDFG.sdfg.arrays):
            raise ValueError(f"Expected that `{nameInside}` is inside the nested SDFG but it was not there (OUTSIDE || {outputNameMapping}).")
    #


    # First we need teh symbol mapping, to properly forward symbols.
    #  If `None` DaCe will try to do this on its own.
    symbol_mapping: Optional[dict[str, Any]] = getNestedSDFGSymbolMapping(parentSDFG, nestedState, translatedNestedSDFG)

    nestedSDFG: dace.SDFG = translatedNestedSDFG.sdfg
    nestedJaxNameMap: dict[str, str] = translatedNestedSDFG.jaxNameMap

    nestedSDFG.validate()
    parentSDFG.validate()

    # Create and add the nested SDFG node
    nestedSDFGNode: NestedSDFG = nestedState.add_nested_sdfg(
            sdfg=nestedSDFG,
            parent=parentSDFG,
            name=name,
            symbol_mapping=symbol_mapping,
            inputs=set(inputNameMapping.values()),      # We need the names inside the nested SDFG
            outputs=set(outputNameMapping.values()),    #  at least I think that.
    )

    # Now we have to create the input and output memlets.
    for nameOutside, nameInside in inputNameMapping.items():
        readNode = nestedState.add_read(nameOutside)
        nestedState.add_edge(readNode, None, nestedSDFGNode, nameInside,
                             dace.Memlet.from_array(nameOutside, parentSDFG.arrays[nameOutside]))
    #
    for nameOutside, nameInside in outputNameMapping.items():
        writeNode = nestedState.add_write(nameOutside)
        nestedState.add_edge(nestedSDFGNode, nameInside, writeNode, None,
                             dace.Memlet.from_array(nameInside, nestedSDFG.arrays[nameInside]))
    #

    # Now propagate the memlets.
    propagate_memlets_sdfg(parentSDFG)

    # Ensure that everything is well
    parentSDFG.validate()

    return nestedSDFGNode
# end def: AddNestedSDFG


