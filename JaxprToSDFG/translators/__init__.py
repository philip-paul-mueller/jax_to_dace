"""This subpackage contains all concrete transformsations 
"""

from JaxprToSDFG.JaxIntrinsicTranslatorInterface    import JaxIntrinsicTranslatorInterface
from .simpleTranslator                              import SimpleTransformator
from .slicingTranslator                             import SlicingTransformator
from .dotGeneralTranslator                          import DotGeneralTranslator

ALL_TRAFOS = [
    SimpleTransformator,
    SlicingTransformator,
    DotGeneralTranslator,
]

