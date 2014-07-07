from __future__ import division

import os
import sys

import numpy as np
from sklearn.externals.six.moves import range
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.testing import assert_almost_equal
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.kernels import MeanMapKernel


################################################################################

def test_mmk():
    bags = [np.random.normal(size=(np.random.randint(10, 100), 10))
            for _ in range(20)]

    res = MeanMapKernel(gamma=2.38).fit_transform(bags)
    for i in range(20):
        for j in range(20):
            exp = pairwise_kernels(bags[j], bags[i], metric='rbf', gamma=2.38)
            assert_almost_equal(res[i, j], exp.mean(),
                                err_msg="({} to {})".format(i, j))

    res = MeanMapKernel(kernel='linear').fit(bags[:5]).transform(bags[-2:])
    for i in range(5):
        for j in range(18, 20):
            exp = pairwise_kernels(bags[j], bags[i], metric='linear')
            assert_almost_equal(res[j - 18, i], exp.mean(),
                                err_msg="({} to {})".format(i, j))

    # fails on wrong dimension
    assert_raises(
        ValueError,
        lambda:MeanMapKernel().fit(bags).transform([np.random.randn(20, 8)]))


################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
