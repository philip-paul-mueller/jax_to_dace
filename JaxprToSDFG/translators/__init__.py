"""This subpackage contains all concrete transformsations 
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .simpleTranslator                              import SimpleTransformator
from .slicingTranslator                             import SlicingTransformator

ALL_TRAFOS = [
    SimpleTransformator,
    SlicingTransformator,
]




