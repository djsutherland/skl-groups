This package contains Cython extensions for the
`skl-groups <https://github.com/dougalsutherland/skl-groups/>`_ package.
Its purpose is to make skl-groups proper easier to install by not requiring
a compiler or having FLANN installed / etc.

It currently contains only a (much) faster implementation of the core portion
of ``skl_groups.divergences.knn.KNNDivergenceEstimator``.


Installation
------------

This code is written and tested for Python 2.7, 3.3, and 3.4. 3.2 probably
works. It is also only tested on OS X and Linux, though it's not known not to
work on other platforms; let me know if you try it.

To install, first install skl-groups,
`cyflann <https://github.com/dougalsutherland/cyflann/>`_,
and a recent version of Cython.
You'll also want to use a compiler with OpenMP support, i.e. gcc instead of
clang (`for now <https://clang-omp.github.io/>`_); if clang is your default,
make sure to set ``CC=gcc`` (or whatever). Then run::

    python setup.py install

to install skl-groups-accel.
