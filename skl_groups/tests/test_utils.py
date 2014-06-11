from __future__ import division

import os
import sys

import numpy as np
from sklearn.externals.six import iteritems
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups import utils


################################################################################

def test_type_utils():
    tests = {
        'bool': (np.array([False, True]), False, True),
        'int32': (np.arange(10, dtype=np.int32), True, True),
        'int64': (np.arange(10, dtype=np.int64), True, True),
        'float32': (np.arange(10, dtype=np.float32), False, False),
        'float64': (np.arange(10, dtype=np.float64), False, False),
    }

    for name, (a, is_int, is_cat) in iteritems(tests):
        assert utils.is_integer_type(a) == is_int, name
        assert utils.is_categorical_type(a) == is_cat, name

        assert utils.is_integer(a[0]) == is_int, name
        assert utils.is_categorical(a[0]) == is_cat, name

    assert utils.is_integer_type(utils.as_integer_type(tests['float32'][0]))
    assert utils.is_integer_type(utils.as_integer_type(tests['float64'][0]))
    assert_raises(
        ValueError, lambda: utils.as_integer_type(tests['float32'][0] + .2))

    assert utils.is_integer(5)
    assert utils.is_categorical(False)
    assert utils.is_categorical(True)

################################################################################

if __name__ == '__main__':
    import nose
    nose.main()