.. _logging:

Progress logging / etc
======================

.. currentmodule:: skl_groups

Some components of skl-groups use the standard Python :mod:`logging` mechanism.
If you want to see this kind of output, you can turn on the logging system.
For example::

    import logging
    logger = logging.getLogger('skl_groups')
    logger.setLevel('INFO')
    logger.addHandler(logging.StreamHandler())

This will show certain steps of the progress output in some potentially slow
algorithms, e.g. :class:`divergences.KNNDivergenceEstimator`. There are also
some warnings given in certain situations.

Some algorithms (currently only :class:`divergences.KNNDivergenceEstimator`)
also have a special "progress" logger, which doesn't propagate up to standard
handlers because it outputs a line at each step through the algorithm.
:func:`utils.show_progress` will set up a
:class:`utils.ProgressBarHandler`, which uses the ``progressbar`` library
(`recommended fork <https://github.com/dougalsutherland/python-progressbar>`_)
to print out pretty console progressbars with ETAs and such::

    from skl_groups.utils import show_progress
    show_progress('skl_groups.divergences.knn.progress')
