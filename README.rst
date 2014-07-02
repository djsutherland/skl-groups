skl-groups
==========

skl-groups is a package to perform machine learning on sets (or "groups") of
features in Python. It extends the `scikit-learn <http://scikit-learn.org>`_
library with support for either transforming sets into feature vectors that
can be operated on with standard scikit-learn constructs or obtaining
pairwise similarity/etc matrices that can be turned into kernels for use in
scikit-learn.

For an introduction to the package, why you might want to use it, and how to
do so, check out
`the documentation <https://dougalsutherland.github.io/skl-groups/>`_.

skl-groups is currently in early stages of development; feel free to get in
touch (dsutherl@cs.cmu.edu) if you're interested.


Installation
------------

Full instructions are
`in the documentation <https://dougalsutherland.github.io/skl-groups/installation.html>`_,
but the short version is to do::

    $ conda install -c http://conda.binstar.org/dougal skl-groups

if you use conda, or::

    $ pip install skl-groups

if not. If you pip install and want to use the kNN divergence estimator,
you'll need to install either
`cyflann <https://github.com/dougalsutherland/cyflann/>`_
or the regular pyflann bindings to FLANN,
and you'll want a version of FLANN with OpenMP support.
A faster version of the kNN estimator is enabled by the
`skl-groups-accel <https://github.com/dougalsutherland/skl-groups-accel>`_
package (also in pip), which requires cyflann
and a working C compiler with OpenMP support (i.e. gcc).
