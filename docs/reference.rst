API reference
=============

.. currentmodule:: skl_groups
.. toctree::


Features representation
-----------------------
.. autosummary::
    :toctree: skl_groups
    :nosignatures:

    features.Features


Preprocessing
-------------

.. autosummary::
    :toctree: skl_groups
    :nosignatures:

    preprocessing.BagMinMaxScaler
    preprocessing.BagNormalizer
    preprocessing.BagPCA
    preprocessing.BagPreprocesser
    preprocessing.BagStandardizer

Summaries
---------

Methods that convert set-based features into single feature vectors.

.. autosummary::
    :toctree: skl_groups
    :nosignatures:
    
    summaries.BagMean
    summaries.BagOfWords
    summaries.L2DensityTransformer


Divergences
-----------

These methods assume feature sets are samples from some probability distribution
and estimate divergences, distances, and so on between those distributions.

.. autosummary::
    :toctree: skl_groups
    :nosignatures:

    divergences.KNNDivergenceEstimator


Kernel utilities
----------------

Utilities to convert generic affinities into valid kernels.

.. autosummary::
    :toctree: skl_groups
    :nosignatures:

    kernels.PairwisePicker
    kernels.Symmetrize
    kernels.ProjectPSD
    kernels.FlipPSD
    kernels.ShiftPSD
    kernels.SquarePSD
