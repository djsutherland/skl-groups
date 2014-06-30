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

The conda install includes skl-groups-accel, described below.

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
(If you want to use the accelerated version, described below, you need cyflann.)

skl-groups is only tested with FLANN 1.8.4.


**Anaconda:**
The skl-groups package pulls this in anyway, but::

    $ conda install -c http://conda.binstar.org/dougal cyflann


**OS X:**
FLANN is available from `Homebrew <http://brew.sh/>`_:
``brew install homebrew/science/flann``.
You can do ``--enable-python`` to get pyflann, but be aware that
Homebrew's interaction with python can be unintuitive:
`see the wiki <https://github.com/Homebrew/homebrew/wiki/Homebrew-and-Python#homebrew-provided-python-bindings>`_.
You can also install cyflann with
``FLANN_DIR=$(brew --prefix) pip install cyflann``.


**Ubuntu:**
``# apt-get install libflann1 flann-dev && FLANN_DIR=/usr pip install cyflann``.
pyflann is not packaged.


**Fedora:**
``# yum install flann flann-devel && FLANN_DIR=/usr pip install cyflann``
or ``# yum install flann-python``.

**CentOS:**
`EPEL <https://fedoraproject.org/wiki/EPEL>`_ has flann packages,
but they're old and probably don't work with skl-groups-accel.
Compile from source.

**Arch:**
The `AUR flann package <https://aur.archlinux.org/packages/flann/>`_ includes pyflann; cyflann wants ``FLANN_DIR=/usr``.


**From source:**
Download `the source <http://www.cs.ubc.ca/research/flann/#download>`_
and follow the `user manual <http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf>`_ to install.
Make sure you're using the OpenMP bindings,
and install the Python bindings and/or run
``FLANN_DIR=$prefix pip install cyflann`` after,
where ``$prefix`` is whatever ``CMAKE_INSTALL_PREFIX`` was (usually ``/usr/local``).


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

or `get the ZIP <https://github.com/dougalsutherland/skl-groups-accel/archive/master.zip>`_
and install it the same way (*after* installing skl-groups).
