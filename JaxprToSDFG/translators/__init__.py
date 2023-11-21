"""This subpackage contains all concrete transformsations 
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .simpleTranslator                              import SimpleTranslator
from .slicingTranslator                             import SlicingTranslator
from .dotGeneralTranslator                          import DotGeneralTranslator
from .broadcastinDimTranslator                      import BroadcastInDimTranslator
from .selectNTranslator                             import SelectNTranslator
from .gatherTranslator                              import GatherTranslator
from .concatenateTranslator                         import ConcatenateTranslator
from .convertElementTypeTranslator                  import ConvertElementTypeTranslator
from .devicePutTranslator                           import DevicePutTranslator

ALL_TRAFOS = [
    SimpleTranslator,
    SlicingTranslator,
    DotGeneralTranslator,
    BroadcastInDimTranslator,
    SelectNTranslator,
    GatherTranslator,
    ConcatenateTranslator,
    ConvertElementTypeTranslator,
    DevicePutTranslator,
]

