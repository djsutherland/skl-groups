Planned improvements
====================

skl-groups is a young package (the
`first commit <https://github.com/dougalsutherland/skl-groups/commit/e0e2013a>`_
was on June 9, 2014, though it's partially based on the previously-existing
`py-sdm <https://github.com/dougalsutherland/py-sdm>`_ package that saw a bit
of real-world use).
Here are some of the things we plan to improve:

* Better support for parameter tuning and cross-validation (high-priority).
* Preconstructed pipelines to do "standard" operations, like
  :class:`skl_groups.divergences.KNNDivergenceEstimator` followed by
  :class:`skl_groups.kernels.PairwisePicker`,
  :class:`skl_groups.kernels.Symmetrize`,
  :class:`skl_groups.kernels.RBFize`,
  :class:`skl_groups.kernels.ProjectPSD`,
  then :class:`sklearn.svm.SVC`,
  with appropriate parameter tuning and so on.
  (A replacement for :class:`sdm.SDC` and friends.)
* Add more algorithms, in particular new KDE-based
  divergence estimators with finite-sample convergence guarantees and
  Fisher vectors.
* Command-line helpers to perform common tasks.
* Better user documentation.
* Everything else on the
  `issues list <https://github.com/dougalsutherland/skl-groups/issues>`_.
