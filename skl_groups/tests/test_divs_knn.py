from __future__ import division

from functools import partial
import logging
import os
import sys

import numpy as np
from scipy.special import psi
from sklearn.externals.six.moves import xrange

from nose.tools import assert_raises
from testfixtures import LogCapture

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups import Features
from skl_groups.divergences import KNNDivergenceEstimator


################################################################################

def test_knn_basic():
    dim = 3
    n = 20
    np.random.seed(47)
    bags = Features([np.random.randn(np.random.randint(30, 100), dim)
                     for _ in xrange(n)])

    # just make sure it runs
    est = KNNDivergenceEstimator(
        div_funcs=('kl', 'js', 'renyi:.9', 'l2'), Ks=(3, 4))
    res = est.fit_transform(bags)
    assert res.shape == (4, 2, n, n, 2)
    assert np.all(np.isfinite(res))

    # test that JS blows up when there's a huge difference in bag sizes
    # (so that K is too low)
    assert_raises(
        ValueError,
        partial(est.fit_transform, bags + [np.random.randn(1000, dim)]))

    # test fit() and then transform() with JS, with different-sized test bags
    est = KNNDivergenceEstimator(div_funcs=('js',), Ks=(5,))
    est.fit(bags)
    with LogCapture('skl_groups.divergences.knn', level=logging.WARNING) as l:
        res = est.transform([np.random.randn(300, dim)])
        assert len(l.records) == 1
        assert l.records[0].message.startswith('Fit with a lower max_K')


def test_knn_kl():
    # verified by hand
    # Dhat(P||Q) = \log m/(n-1) + d / n  \sum_{i=1}^n \log \nu_k(i)/rho_k(i)
    x = np.reshape([0., 1, 3], (3, 1))
    y = np.reshape([.2, 1.2, 3.2, 7.2], (4, 1))

    n = x.shape[0]
    m = y.shape[0]

    x_to_y = np.log(m / (n-1)) + 1/n * (
        np.log(1.2 / 3) + np.log(.8 / 2) + np.log(1.8 / 3))
    y_to_x = np.log(n / (m-1)) + 1/m * (
        np.log(.8 / 3) + np.log(1.2 / 2) + np.log(2.2 / 3) + np.log(6.2 / 6))

    est = KNNDivergenceEstimator(div_funcs=['kl'], Ks=[2], clamp=False)
    res = est.fit_transform([x, y]).squeeze()
    # assert res[0, 0] == 0
    # assert res[1, 1] == 0
    assert np.allclose(res[0, 1], x_to_y), "{} vs {}".format(res[0, 1], x_to_y)
    assert np.allclose(res[1, 0], y_to_x), "{} vs {}".format(res[1, 0], y_to_x)


def test_knn_js():
    # verified by hand
    x = np.reshape([0, 1, 3], (3, 1))
    y = np.reshape([.2, 1.2, 3.2, 6.2], (4, 1))

    mix_ent = np.log(2) + np.log(3) + psi(2) \
        + (np.log(.2) + np.log(.8) + np.log(1.8) - psi(1) - 2*psi(2)) / 6 \
        + (np.log(.2) + np.log(2) + np.log(3.2) - psi(1) - 3*psi(2)) / 8
    x_ent = np.log(2) + (np.log(3) + np.log(2) + np.log(3)) / 3
    y_ent = np.log(3) + (np.log(3) + np.log(2) + np.log(3) + np.log(5)) / 4
    right_js = mix_ent - (x_ent + y_ent) / 2
    # TODO: clamping???

    est = KNNDivergenceEstimator(div_funcs=['js'], Ks=[2])
    res = est.fit([x]).transform([y]).squeeze()
    assert res.shape == (2,)
    assert res[0] == res[1]
    err_msg = "got {}, expected {}"
    assert np.allclose(res[0], right_js), err_msg.format(res, right_js)



################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
