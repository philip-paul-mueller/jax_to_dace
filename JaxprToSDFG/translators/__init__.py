"""This subpackage contains all concrete transformsations 
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .simpleTranslator                              import SimpleTransformator
from .slicingTranslator                             import SlicingTransformator
from .broadcastinDimTranslator                      import BroadcastInDimTransformator
from .selectNTranslator                             import SelectNTransformator
from .gatherTranslator                              import GatherTransformator

ALL_TRAFOS = [
    SimpleTransformator,
    SlicingTransformator,
    BroadcastInDimTransformator,
    SelectNTransformator,
    GatherTransformator,
]




