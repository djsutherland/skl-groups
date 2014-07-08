from __future__ import division, print_function

import logging
import json
import sys

import numpy as np
from sklearn.externals import six


def identity(x):
    return x


################################################################################
### Type utilities for numpy

def is_integer_type(x):
    "Checks whether the array is of an integral type."
    return issubclass(np.asanyarray(x).dtype.type, np.integer)


def is_categorical_type(ary):
    "Checks whether the array is either integral or boolean."
    ary = np.asanyarray(ary)
    return is_integer_type(ary) or ary.dtype.kind == 'b'


def is_integer(x):
    "Checks whether the argument is a single integer."
    return np.isscalar(x) and is_integer_type(x)


def is_categorical(x):
    "Checks whether the argument is a single integer or boolean."
    return np.isscalar(x) and is_categorical_type(x)


def as_integer_type(ary):
    '''
    Returns argument as an integer array, converting floats if convertable.
    Raises ValueError if it's a float array with nonintegral values.
    '''
    ary = np.asanyarray(ary)
    if is_integer_type(ary):
        return ary
    rounded = np.rint(ary)
    if np.any(rounded != ary):
        raise ValueError("argument array must contain only integers")
    return rounded.astype(int)


################################################################################

class NoImportStub(object):
    '''
    A class to use so that if there are missing optional dependencies, we can
    still import the name, but using it will raise the original import error
    that prevented us using the name anyway.
    '''
    def __init__(self, exception):
        self.exception = exception

    def __getattribute__(self, name):
        if name == 'exception':
            return object.__getattribute__(self, name)
        six.reraise(*self.exception)

    def __call__(self, *args, **kwargs):
        six.reraise(*self.exception)


################################################################################

class ProgressLogger(object):
    '''
    A little class to use to wrap logging progress to a logger object.

    Parameters
    ----------
    logger : :class:`logging.Logger`
        The logger instance to use.

    name : string, optional, default None
        The name of the process.

    Notes
    -----
    Instances of this class can be used as callables. In that case,
    they take an iterable and return a generator that yields each element, with
    appropriate progress messages in between. For example::

        ProgressLogger(logger)([x ** 2 for x in range(10)])
        ProgressLogger(logger)((x ** 2 for x in range(10)), total=10)

    will each yield the integers 1, 4, 9, ... while logging progress messages
    to ``logger``. In the second case, we passed the ``total`` argument to
    communicate the length of the iterable, which is not available via
    :func:`len`.

    A gotcha here is that :meth:`finish` will only be called if the iterator
    actually terminates. For example::

        for x, y in zip(things, ProgressLogger(logger)(f(x) for x in things)):
            pass

    won't ever :meth:`finish`, because :func:`zip` will see that ``things``
    ends first and not continue iterating over the generator.
    '''
    def __init__(self, logger, name=None):
        self.logger = logger
        self.name = name

    def start(self, total):
        '''
        Signal the start of the process.

        Parameters
        ----------
        total : int
            The total number of steps in the process, or None if unknown.
        '''
        self.logger.info(json.dumps(['START', self.name, total]))

    def update(self, idx):
        '''
        Update the current state.

        Parameters
        ----------
        idx : int
            The current state through the process.
        '''
        # json format, but since we might call this a lot do it a little faster
        self.logger.info('["SET", {}]'.format(idx))

    def finish(self):
        '''
        Marks the process as done.
        '''
        self.logger.info(json.dumps(['DONE']))

    def __call__(self, it, total=None):
        if total is None:
            try:
                total = len(it)
            except TypeError:
                total = None
        self.start(total)
        for i, thing in enumerate(it, 1):
            self.update(i)
            yield thing
        self.finish()


class ProgressBarHandler(logging.Handler):
    '''
    A logging handler that uses the progressbar module to show progress from
    a :class:`ProgressLogger`.

    Takes the same parameters as :class:`progressbar.ProgressBar`,
    but gives a default for ``widgets`` that applies only when maxval is
    available; you'll need to pass different widgets if not.
    '''
    def __init__(self, widgets=None, **kwargs):
        import progressbar as pb

        logging.Handler.__init__(self)

        if widgets is None:
            class CommaProgress(pb.Widget):
                def update(self, pbar):
                    return '{:,} of {:,}'.format(pbar.currval, pbar.maxval)

            widgets = [' ', CommaProgress(), ' (', pb.Percentage(), ') ',
                       pb.Bar(), ' ', pb.ETA()]

        self.pbar_args = {'widgets': widgets}
        self.pbar_args.update(kwargs)

    def emit(self, record):
        import progressbar as pb

        msg = json.loads(record.msg)
        # print(msg)
        if msg[0] == 'SET':
            pass
            self.pbar.update(msg[1])
        elif msg[0] == 'START':
            print(msg[1] + ':', file=sys.stderr)
            self.pbar = pb.ProgressBar(maxval=msg[2], **self.pbar_args)
            self.pbar.start()
        elif msg[0] == 'DONE':
            self.pbar.finish()
            del self.pbar
            print('', file=sys.stderr)


def show_progress(name, **kwargs):
    '''
    Sets up a :class:`ProgressBarHandler` to handle progess logs for
    a given module.

    Parameters
    ----------
    name : string
        The module name of the progress logger to use. For example,
        :class:`skl_groups.divergences.KNNDivergenceEstimator`
        uses ``'skl_groups.divergences.knn.progress'``.

    * : anything
        Other keyword arguments are passed to the :class:`ProgressBarHandler`.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(ProgressBarHandler(**kwargs))
