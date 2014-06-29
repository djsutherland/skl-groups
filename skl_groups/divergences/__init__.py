'''
Methods that assume feature sets are samples from some probability distribution
and estimate divergences, distances, and so on between those distributions.
'''
from ..utils import NoImportStub

try:
    from .knn import KNNDivergenceEstimator
except ImportError as e:
    if hasattr(e, 'message'):  # python 2
        msg = e.message
    else:  # python 3
        msg = e.msg
    KNNDivergenceEstimator = NoImportStub(msg)
