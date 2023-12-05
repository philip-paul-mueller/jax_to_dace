"""Implements the squeeze translator.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import dace
from typing import Union


class SqueezeTranslator(JaxIntrinsicTranslatorInterface):
    """Allows to remove dimensions with size one.

    Essentially equivalent to `np.squeeze`.
    There are two different modes that are supported, either using a memlet (the default) or using a copy map.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a slicing translators
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
        return str(eqn.primitive.name) == "squeeze"
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
        use_map = False

        if(len(eqn.invars) != 1):
            raise ValueError(f"Squeezing only supports one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(len(eqn.invars[0].aval.shape) != len(eqn.outvars[0].aval.shape) + len(eqn.params['dimensions'])):
            raise ValueError("Expected that the input and output have the same numbers of dimensions.")
        if(any([inVarNames[i] is None  for i in range(len(inVarNames))])):
            raise ValueError(f"Does not allow for literals in the input arguments.")
        #

        inAVal  = eqn.invars[0].aval            # The abstract value
        inShape = inAVal.shape                  # The shape of the inpt value.
        outAVal = eqn.outvars[0].aval
        outShape = outAVal.shape

        # These are the dimensions to remove.
        dims_to_remove = eqn.params['dimensions']

        if(not all([inShape[dtr] == 1  for dtr in dims_to_remove])):   
            raise ValueError(f"Strange some dimensions that with a different length than one should be removed, shoudl remove: {[(dim, inShape[dtr])  for dim, dtr in enumerate(dims_to_remove)]}")
        #

        if(use_map):
            # The map ranges will go through all the output dimensions.
            tMapRanges = []
            for dim, dim_size in enumerate(outShape):
                tMapRanges.append( (f'__i{dim}', f'0:{dim_size}') )
            #

            # The Outputs are also just iterating through everything.
            tOutputs_ = []
            for it, _ in tMapRanges:
                tOutputs_.append(it)
            #

            # The inputs are a bit different, but also very simple
            tInputs_ = []
            itCnt = 0
            for dim in range(len(inShape)):
                if(dim in dims_to_remove):
                    tInputs_.append('0')        # Only valid index
                else:
                    tInputs_.append(tMapRanges[itCnt][0])         # This dimension must have the same size.
                    itCnt += 1
                assert itCnt <= len(tMapRanges)
            assert itCnt == len(tMapRanges)

            tName = f'_squeeze_{outVarNames[0]}'
            tCode = '__out = __in'

            eqnState.add_mapped_tasklet(
                name=tName,
                map_ranges={k:v  for k, v in tMapRanges},
                inputs={'__in': dace.Memlet.simple(inVarNames[0], ', '.join(tInputs_))},
                code=tCode,
                outputs={'__out': dace.Memlet.simple(outVarNames[0], ', '.join(tOutputs_))},
                external_edges=True,
            )

        else:
            # Here we are using a memlet directly.
            tInputs_, tOutputs_ = [], []

            for dim, dim_size in enumerate(inShape):
                tInputs_.append( f'0:{dim_size}' )      # The output is always present.
                if(dim not in dims_to_remove):
                    tOutputs_.append( tInputs_[-1] )
            #

            inAN   = eqnState.add_read(inVarNames[0])
            outAN  = eqnState.add_write(outVarNames[0])
            memlet = dace.Memlet(
                        inVarNames[0],
                        subset=', '.join(tInputs_),
                        other_subset=', '.join(tOutputs_),
            )
            eqnState.add_nedge(inAN, outAN, memlet)
        #

        return eqnState
    # end def: translateEqn

# end class(SqueezeTranslator):

