"""This implements the `broadcast_in_dim` intrinsic.
"""
from JaxprToSDFG.JaxIntrinsicTranslatorInterface import JaxIntrinsicTranslatorInterface

from jax._src.core import JaxprEqn
import numpy as np
import dace
from dace import subsets
from typing import Union
from math import prod


class BroadcastInDimTranslator(JaxIntrinsicTranslatorInterface):
    """This handles the `broadcast_in_dim` intrinsic.

    It requieres two parameters `shape` and `broadcast_dimensions`, the following is taken from the jax source:
    - shape:
        The shape of the target array
    - broadcast_dimensions:
        To which dimension in the target shape each dimension of the operand shape corresponds to.
        That is, dimension i of the operand becomes dimension broadcast_dimensions[i] of the result.

    Notes:
        It seams that Jax maps slicing with a non one step size to a combination of the `broadcast_in_dim` and `gather`.
    """
    __slots__ = ()


    def __init__(self):
        """Initializes a `briadcast_in_dim` translators
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
        return str(eqn.primitive.name) == "broadcast_in_dim"
    # end def: canHandle


    def translateEqn(self,
                     translator,
                     inVarNames: list[Union[str, None]],
                     outVarNames: list[str],
                     eqn: JaxprEqn,
                     eqnState: dace.SDFGState,
    ):
        """Translate eqn into an SDFG that is created inside `eqnState`.

        In essence it creates a copy of the array this is how slicing is implemented.
        We basically hope that DaCe is able to exploit that.
        It could also be made into a view.

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
        outShape     = eqn.outvars[0].aval.shape
        shape        = eqn.params['shape'               ]
        bDims        = eqn.params['broadcast_dimensions']
        inpIsLiteral = False
        inpIsScalar  = False

        if(inVarNames[0] is None):
            assert bDims ==  ()
            inpIsLiteral = True
            inShape      = ()

        elif((eqn.invars[0].aval.shape == ()) and (isinstance(translator.getArray(inVarNames[0]), dace.data.Scalar))):
            # We have a scalar and we will handle it the same as we would do a literal.
            inpIsScalar = True
            inShape     = ()

        else:
            inShape  = eqn.invars[0].aval.shape
            assert len(bDims) > 0, f"Expected to find an array, but found a scalarish thing."
        #

        if(len(eqn.invars) != 1):
            raise ValueError(f"`broadcast_in_dim` only supports one input argument.")
        if(len(eqn.outvars) != 1):
            raise ValueError(f"Expected only one return value of equation '{str(eqn)}' but it had {len(eqn.outvars)}")
        if(eqn.outvars[0].aval.shape != shape):
            raise ValueError(f"Parameters specified a shape of `{shape}` but the output variable ad a shape of `{eqn.outvars[0].aval.shape}`.")
        if(outShape != shape):
            raise ValueError(f"shape of the output was `{outShape}`, but the specified shape was `{shape}`.")
        if(len(bDims) == 0  and  (inpIsLiteral or inpIsScalar)):
            pass            # Essentially 'numpy.full()'
        elif(prod(inShape) == prod(outShape)):
            pass            # essentially this should be a reshape (more test would be fine)
        elif(len(bDims) == 0  and  (not inpIsLiteral)):
            raise ValueError(f"No broadcast dimension specified.")
        elif(len(bDims) == 1):
            pass    # No test for sorting
        elif(not all([bDims[i-1] < bDims[i]  for i in range(1, len(bDims))])):    
            raise ValueError(f"This function assumes that `broadcast_dimensions` is sorted but it was not (`{bDims}`)")
        #

        if((not (inpIsLiteral or inpIsScalar)) and ((len(bDims) == 1) or (prod(inShape) == prod(outShape)))):
            # The code is inspired from the `reshape()` function of the python frontend
            # TODO: Probably poting to view, but this allows to bypass aliasing.
            inName    = inVarNames[0]
            outName   = outVarNames[0]
            inArr     = translator.getArray(inName)
            outArr    = translator.getArray(outName)
            inSet     = subsets.Range.from_array(inArr)
            outSet    = subsets.Range.from_array(outArr)
            readNode  = eqnState.add_read(inName)
            writeNode = eqnState.add_write(outName)
            memlet    = dace.Memlet(data=inName, subset=inSet, other_subset=outSet)

            eqnState.add_nedge(readNode, writeNode, data=memlet)

        elif((not (inpIsLiteral or inpIsScalar)) and (len(bDims) != 1)):
            raise NotImplementedError(f"Multiple broadcast dimensions are not implemented.")

        elif(inpIsLiteral or inpIsScalar):
            # This code is inspired from the `numpy.full` function.
            inName    = inVarNames[0]
            outName   = outVarNames[0]
            outArr    = translator.getArray(outName)

            if(inpIsScalar):
                inputs = {'__in_scalar': dace.Memlet(data=inName, subset='0')}
                tVal   = '__in_scalar'
            elif(inpIsLiteral):
                inputs = dict()
                jaxInVar  = eqn.invars[0]
                tVal = jaxInVar.val
                if(isinstance(tVal, np.ndarray)):
                    tVal = jaxInVar.val.max()
                #
            else:
                raise NotImplementedError("What are you doing man.")
            #

            eqnState.add_mapped_tasklet(
                '_fill_',
                map_ranges={f"__i{dim}": f"0:{s}" for dim, s in enumerate(outShape)},
                inputs=inputs,
                code=f"__out = {tVal}",
                outputs={'__out': dace.Memlet.simple(outName, ",".join([f"__i{dim}" for dim in range(len(outShape))]))},
                external_edges=True
            )
        else:
            raise NotImplementedError(f"This case is not implemented.")
        #

        return eqnState
    # end def: translateEqn

# end class(BroadcastInDimTranslator):

