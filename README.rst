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

This code is written and tested for Python 2.7, 3.3, and 3.4. 3.2 probably
works. It is also only tested on OS X and Linux, though it's not known not to
work on other platforms; let me know if you try it.

Get reasonably recent versions of numpy, scipy, and scikit-learn, then run::

    python setup.py install

to install skl-groups. Once the package is a little more mature, it will also
be available through pip or conda, but not just yet.

If you want to use the kNN divergence estimator, you'll need to have
`FLANN <http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN>`_
and either its Python interface (pyflann) or the
`cyflann <https://github.com/dougalsutherland/cyflann/>`_ library.
These can be installed after installing skl-groups.
If you're running on non-tiny datasets, you'll want a version with OpenMP
support.
If you use Anaconda, the cyflann page has a pointer on how to install both
FLANN and cyflann through conda.

You may also want the
`skl-groups-accel <https://github.com/dougalsutherland/skl-groups-accel>`_
package, which contains faster implementations of some of the algorithms in
skl-groups; it's split out to make the base package of skl-groups easier to
install. (Currently, it only has the kNN divergence estimator).
