"""This subpackage contains all concrete transformsations 

If you write a new translator, import it here and add it to the `ALL_TRAFOS`.
It will then automatically created by an `JaxprToSDFG` instance.
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .ALUTranslator                                 import ALUTranslator
from .slicingTranslator                             import SlicingTranslator
from .dotGeneralTranslator                          import DotGeneralTranslator
from .reductionTranslator                           import ReductionTranslator
from .reshapeTranslator                             import ReshapeTranslator
from .broadcastinDimTranslator                      import BroadcastInDimTranslator
from .selectNTranslator                             import SelectNTranslator
from .gatherTranslator                              import GatherTranslator
from .concatenateTranslator                         import ConcatenateTranslator
from .convertElementTypeTranslator                  import ConvertElementTypeTranslator
from .devicePutTranslator                           import DevicePutTranslator
from .pjitTranslator                                import PJITTranslator
from .squeezeTranslator                             import SqueezeTranslator
from .iotaTranslator                                import IotaTranslator

ALL_TRAFOS = [
    ALUTranslator,
    SlicingTranslator,
    DotGeneralTranslator,
    ReductionTranslator,
    ReshapeTranslator,
    BroadcastInDimTranslator,
    SelectNTranslator,
    GatherTranslator,
    ConcatenateTranslator,
    ConvertElementTypeTranslator,
    DevicePutTranslator,
    PJITTranslator,
    SqueezeTranslator,
    IotaTranslator,
]

