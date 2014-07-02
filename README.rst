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

The easy way to install is::

    $ conda install -c http://conda.binstar.org/dougal skl-groups

if you use conda, which includes both skl-groups and skl-groups-accel.

Otherwise, first install skl-groups and
`cyflann <https://github.com/dougalsutherland/cyflann/>`_.
You'll also need a compiler with OpenMP support,
so gcc (not llvm-gcc) instead of clang
(`for now <https://clang-omp.github.io/>`_).
On OSX, do ``brew install gcc`` and then set ``CC=gcc-4.8``;
Linux users probably have gcc installed,
but make sure to set ``CC`` if you default to clang.
Then you can::

    $ pip install skl-groups-accel

or, from a source checkout,
``python setup.py install`` or ``pip install .`` or whatever.
If you're installing from a source checkout you need a recent version of Cython.
