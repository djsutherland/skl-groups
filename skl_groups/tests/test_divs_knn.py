from __future__ import division

from functools import partial
import logging
import os
import sys
import tempfile

import numpy as np
from scipy.special import psi
from sklearn.externals.six.moves import xrange, zip

from nose.tools import assert_raises
from sklearn.utils.testing import assert_array_almost_equal
from nose import SkipTest
from testfixtures import LogCapture

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.features import Features


have_flann = True
try:
    import cyflann
except ImportError:
    try:
        import pyflann
    except ImportError:
        have_flann = False

if have_flann:
    from skl_groups.divergences import KNNDivergenceEstimator

    try:
        import skl_groups_accel.knn_divs
    except ImportError:
        have_accel = False
    else:
        have_accel = True


################################################################################

def test_knn_version_consistency():
    if not have_flann:
        raise SkipTest("No flann, so skipping knn tests.")
    if not have_accel:
        raise SkipTest("No skl-groups-accel, so skipping version consistency.")

    n = 20
    for dim in [1, 7]:
        np.random.seed(47)
        bags = Features([np.random.randn(np.random.randint(30, 100), dim)
                         for _ in xrange(n)])

        div_funcs = ('kl', 'js', 'renyi:.9', 'l2', 'tsallis:.8')
        Ks = (3, 4)
        get_est = partial(KNNDivergenceEstimator, div_funcs=div_funcs, Ks=Ks)
        results = {}
        for version in ('fast', 'slow', 'best'):
            est = get_est(version=version)
            results[version] = res = est.fit_transform(bags)
            assert res.shape == (len(div_funcs), len(Ks), n, n)
            assert np.all(np.isfinite(res))

        for df, fast, slow in zip(div_funcs, results['fast'], results['slow']):
            assert_array_almost_equal(
                fast, slow, decimal=1 if df == 'js' else 5,
                err_msg="({}, dim {})".format(df, dim))
            # TODO: debug JS differences

        est = get_est(version='fast', n_jobs=-1)
        res = est.fit_transform(bags)
        assert np.all(results['fast'] == res)

        est = get_est(version='slow', n_jobs=-1)
        res = est.fit_transform(bags)
        assert np.all(results['slow'] == res)


def test_knn_sanity_slow():
    if not have_flann:
        raise SkipTest("No flann, so skipping knn tests.")

    dim = 3
    n = 20
    np.random.seed(47)
    bags = Features([np.random.randn(np.random.randint(30, 100), dim)
                     for _ in xrange(n)])

    # just make sure it runs
    div_funcs = ('kl', 'js', 'renyi:.9', 'l2', 'tsallis:.8')
    Ks = (3, 4)
    est = KNNDivergenceEstimator(div_funcs=div_funcs, Ks=Ks)
    res = est.fit_transform(bags)
    assert res.shape == (len(div_funcs), len(Ks), n, n)
    assert np.all(np.isfinite(res))

    # test that JS blows up when there's a huge difference in bag sizes
    # (so that K is too low)
    assert_raises(
        ValueError,
        partial(est.fit_transform, bags + [np.random.randn(1000, dim)]))

    # test fit() and then transform() with JS, with different-sized test bags
    est = KNNDivergenceEstimator(div_funcs=('js',), Ks=(5,))
    est.fit(bags, get_rhos=True)
    with LogCapture('skl_groups.divergences.knn', level=logging.WARNING) as l:
        res = est.transform([np.random.randn(300, dim)])
        assert res.shape == (1, 1, 1, len(bags))
        assert len(l.records) == 1
        assert l.records[0].message.startswith('Y_rhos had a lower max_K')

    # test that passing div func more than once raises
    def blah(df):
        est = KNNDivergenceEstimator(div_funcs=[df, df])
        return est.fit(bags)
    assert_raises(ValueError, lambda: blah('kl'))
    assert_raises(ValueError, lambda: blah('renyi:.8'))
    assert_raises(ValueError, lambda: blah('l2'))


def test_knn_memory():
    if not have_flann:
        raise SkipTest("No flann, so skipping knn tests.")

    dim = 3
    n = 20
    np.random.seed(47)
    bags = Features([np.random.randn(np.random.randint(30, 100), dim)
                     for _ in xrange(n)])

    tdir = tempfile.mkdtemp()
    div_funcs = ('kl', 'js', 'renyi:.9', 'l2', 'tsallis:.8')
    Ks = (3, 4)
    est = KNNDivergenceEstimator(div_funcs=div_funcs, Ks=Ks, memory=tdir)
    res1 = est.fit_transform(bags)

    with LogCapture('skl_groups.divergences.knn', level=logging.INFO) as l:
        res2 = est.transform(bags)
        assert len(l.records) == 0
    assert np.all(res1 == res2)

    with LogCapture('skl_groups.divergences.knn', level=logging.INFO) as l:
        res3 = est.fit_transform(bags)
        for r in l.records:
            assert not r.message.startswith("Getting divergences")
    assert np.all(res1 == res3)


def test_knn_kl():
    if not have_flann:
        raise SkipTest("No flann, so skipping knn tests.")

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

    msg = "got {}, expected {}"
    est = KNNDivergenceEstimator(div_funcs=['kl'], Ks=[2], clamp=False)
    res = est.fit_transform([x, y]).squeeze()
    assert res[0, 0] == 0
    assert res[1, 1] == 0
    assert np.allclose(res[0, 1], x_to_y), msg.format(res[0, 1], x_to_y)
    assert np.allclose(res[1, 0], y_to_x), msg.format(res[1, 0], y_to_x)


def test_knn_js():
    if not have_flann:
        raise SkipTest("No flann, so skipping knn tests.")

    # verified by hand
    x = np.reshape([0., 1, 3, 6], (4, 1))
    n = 4

    y = np.reshape([.2, 1.2, 3.2, 6.2, 10.2], (5, 1))
    m = 5

    M = 2

    right_js = (
        np.log(n + m - 1) + psi(M)
        + 1/(2*n) * (  # x weight is 1/7, y weight is 4/35, quantile 1/4
            np.log(.2) - psi(1)     # 0 => .2(y), 1(x)
            + np.log(.8) - psi(2)   # 1 => 1.2(y), .2(y)
            + np.log(1.8) - psi(2)  # 3 => 3.2(y), 1.2(y)
            + np.log(2.8) - psi(2)  # 6 => 6.2(y), 3.2(y)
        )
        + 1/(2*m) * (  # x weight is 5/36, y weight is 1/9, quantile 1/4
            np.log(.2) - psi(1)     # .2 => 0(x)
            + np.log(1) - psi(2)    # 1.2 => 1(x), .2(y)
            + np.log(2) - psi(2)    # 3.2 => 3(x), 1.2(y)
            + np.log(3) - psi(2)    # 6.2 => 6(x), 3.2(y)
            + np.log(4.2) - psi(2)  # 10.2 => 6.2(y), 6(x)
        )
        - 1/2 * np.log(n-1) - 1/(2*n) * (
            np.log(3) + np.log(2) + np.log(3) + np.log(5))
        - 1/2 * np.log(m-1) - 1/(2*m) * (
            np.log(3) + np.log(2) + np.log(3) + np.log(4) + np.log(7))
    )

    msg = "got {}, expected {}"
    est = KNNDivergenceEstimator(div_funcs=['js'], Ks=[2], clamp=False)
    res = est.fit([x]).transform([y])
    assert res.shape == (1, 1, 1, 1)
    res = res[0, 0, 0, 0]
    assert np.allclose(res, right_js, atol=1e-6), msg.format(res, right_js)



################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
