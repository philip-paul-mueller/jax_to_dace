"""This subpackage contains all concrete transformsations 
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .simpleTranslator                              import SimpleTransformator
from .slicingTranslator                             import SlicingTransformator
from .broadcastinDimTranslator                      import BroadcastInDimTransformator

ALL_TRAFOS = [
    SimpleTransformator,
    SlicingTransformator,
    BroadcastInDimTransformator,
]




