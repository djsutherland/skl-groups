from __future__ import division, print_function

import numpy as np


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
    if is_integer_type(ary):
        return ary
    rounded = np.rint(ary)
    if np.any(rounded != ary):
        raise ValueError("argument array must contain only integers")
    return rounded.astype(int)
