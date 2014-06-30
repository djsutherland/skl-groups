Installation
============

First, you'll want a Python installation
(we test with versions 2.7, 3.3, and 3.4; 3.2 probably works),
and reasonably recent versions of 
`numpy <http://numpy.org>`_,
`scipy <http://scipy.org>`_,
and `scikit-learn <http://scikit-learn.org>`_.
If you're not already attached to a particular install of Python,
we recommend the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
distribution, which makes life easy;
otherwise, `these instructions <http://scikit-learn.org/stable/install.html>`_
might be helpful.
Note that skl-groups is not currently tested on Windows (or FreeBSD, or anything
other than recent versions of OSX, Arch, and CentOS). If you try it, let us
know.


Release versions
----------------

The released version of skl-groups is available via pypi::

    $ pip install skl-groups

or, if you use Anaconda, via my binstar channel::

    $ conda install -c http://conda.binstar.org/dougal skl-groups

At this point, there's nothing special about the releases other than that
they're easier to install.


Development version
-------------------

You can install a development version by getting the code from
`the repository <https://github.com/dougalsutherland/skl-groups/issues>`_,
either with::

    $ git clone https://github.com/dougalsutherland/skl-groups.git

or by downloading and extracting the latest version 
`as a ZIP <https://github.com/dougalsutherland/skl-groups/archive/master.zip>`_.
Then just run::

    $ python setup.py install

to install it. You might need to add ``--user`` to install into your home
directory, or run it with ``sudo``, depending on your setup.

If you chose to clone the repository, you can update the code later by running
``git pull`` and then ``python setup.py install`` again.


FLANN
-----

If you want to use :class:`skl_groups.divergences.KNNDivergenceEstimator`,
you'll also need a Python interface to
`FLANN <http://www.cs.ubc.ca/research/flann/>`_
(the Fast Library for Approximate Nearest Neighbors),
either the ``pyflann`` library included with FLANN or 
`cyflann <https://github.com/dougalsutherland/cyflann/>`_.
If you're using Anaconda, the best way to get it is::

    $ conda install -c http://conda.binstar.org/dougal cyflann

(which is included as a requirement in the skl-groups release package).

Otherwise, check your package manager or install it from source.


Accelerated version
-------------------

Some algorithms
(currently just :class:`skl_groups.divergences.KNNDivergenceEstimator`)
have faster C implementations in the
`skl-groups-accel <https://github.com/dougalsutherland/skl-groups-accel/>`_
package, which is kept separate to make the "base" skl-groups easier to install.

If you're using the conda package, skl-groups-accel is actually included in
skl-groups, because requirements are so much easier in conda-land, and you
don't have to do anything.

Otherwise, skl-groups-accel requires
`cyflann <https://github.com/dougalsutherland/cyflann/>`_
(pyflann is insufficient – in fact, this code is why cyflann was written)
and a recent version of `cython <http://cython.org/>`_
as well as a functioning C compiler with OpenMP support
(i.e. gcc, not clang, `for now <http://clang-omp.github.io>`_;
you may need to ``export CC=gcc`` or similar).

Once you have that, you can get a release version with::

    $ pip install skl-groups-accel

or install the development version the same way as before, getting the code with::

    $ git clone https://github.com/dougalsutherland/skl-groups-accel.git

or `as a ZIP <https://github.com/dougalsutherland/skl-groups/archive/master.zip>`_
and install it the same way (*after* installing skl-groups).
