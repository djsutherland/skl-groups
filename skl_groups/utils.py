from __future__ import division, print_function

import json

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
    "A little class to use to wrap logging progress to a logger object."
    def __init__(self, logger, name=None):
        self.logger = logger
        self.name = name

    def start(self, total):
        self.logger.info(json.dumps(['START', self.name, total]))

    def update(self, idx):
        # json format, but since we might call this a lot do it a little faster
        self.logger.info('["SET", {}]'.format(idx))

    def finish(self):
        self.logger.info(json.dumps(['DONE']))

    def __call__(self, it, total=None):
        if total is None:
            try:
                total = len(it)
            except TypeError:
                total = None
        self.start(total)
        for i, thing in enumerate(it):
            self.update(i)
            yield thing
        self.finish()
