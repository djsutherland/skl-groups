from ..utils import NoImportStub

try:
    from .knn import KNNDivergenceEstimator
except ImportError as e:
    if hasattr(e, 'message'):  # python 2
        msg = e.message
    else:  # python 3
        msg = e.msg
    KNNDivergenceEstimator = NoImportStub(msg)
