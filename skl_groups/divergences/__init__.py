'''
Methods that assume feature sets are samples from some probability distribution
and estimate divergences, distances, and so on between those distributions.
'''
from ..utils import NoImportStub

try:
    from .knn import KNNDivergenceEstimator
except ImportError as e:
    import sys
    KNNDivergenceEstimator = NoImportStub(sys.exc_info())
