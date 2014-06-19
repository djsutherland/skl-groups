from __future__ import division

import os
import sys

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals.six.moves import xrange
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.features import Features
from skl_groups.preprocessing import BagMinMaxScaler
from skl_groups.summaries import BagOfWords, BagMean, L2DensityTransformer


################################################################################

def test_mean():
    dim = 5
    n_bags = 50
    np.random.seed(42)
    bags = [np.random.randn(np.random.randint(30, 100), dim)
            for _ in xrange(n_bags)]

    meaned = BagMean().fit_transform(bags)
    assert meaned.shape == (n_bags, dim)
    assert np.allclose(meaned[3], np.mean(bags[3], axis=0))


def test_bagofwords_basic():
    n_codewords = 10
    dim = 5
    kmeans = KMeans(n_clusters=n_codewords, max_iter=100, n_init=3,
                    random_state=47)
    bow = BagOfWords(kmeans)

    np.random.seed(42)
    bags = [np.random.randn(np.random.randint(30, 100), dim)
            for _ in xrange(50)]

    bowed = bow.fit_transform(bags)
    assert bowed.shape == (len(bags), n_codewords)
    assert bow.codewords_.shape == (n_codewords, dim)
    assert np.all(bowed >= 0)
    assert np.all(np.sum(bowed, 1) == [b.shape[0] for b in bags])

    bow.fit(Features(bags))
    bowed2 = bow.transform(bags)
    assert np.all(bowed == bowed2)
    assert bow.codewords_.shape == (n_codewords, dim)

    minikmeans = MiniBatchKMeans(n_clusters=n_codewords, max_iter=100,
                                 random_state=47)
    minibow = BagOfWords(minikmeans)
    assert_raises(AttributeError, lambda: minibow.transform(bags))
    minibowed = minibow.fit_transform(bags)
    assert minibowed.shape == bowed.shape
    assert np.all(bowed >= 0)
    assert np.all(np.sum(bowed, 1) == [b.shape[0] for b in bags])


def test_l2density_basic():
    dim = 3
    bags = [np.random.randn(np.random.randint(30, 100), dim)
            for _ in xrange(50)]
    pipe = Pipeline([
        ('scale', BagMinMaxScaler([0, 1])),
        ('density', L2DensityTransformer(15)),
    ])
    l2ed = pipe.fit_transform(bags)

    assert np.all(np.isfinite(l2ed))
    # ||x - y||^2 = <x, x> - 2 <x, y> + <y, y>
    K = l2ed.dot(l2ed.T)
    row_norms_sq = np.diagonal(K)
    l2_dist_sq = row_norms_sq[:, None] - 2 * K + row_norms_sq[None, :]
    assert np.min(row_norms_sq) > 0
    assert np.min(l2_dist_sq) >= 0

    assert_raises(ValueError, lambda: L2DensityTransformer(10, basis='foo'))

    t = L2DensityTransformer(10)
    assert_raises(AttributeError, lambda: t.transform(bags))
    t.fit(dim)
    t.transform(BagMinMaxScaler([0, 1]).fit_transform(bags))
    assert_raises(ValueError, lambda: t.transform([b[:, :2] for b in bags]))
    assert_raises(ValueError, lambda: t.transform(bags))
    t.basis = 'haha snuck my way in'
    assert_raises(ValueError, lambda: t.transform(bags))


################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
