from __future__ import division

import os
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.six.moves import xrange
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.features import Features
from skl_groups.preprocessing import (BagPreprocesser, BagStandardizer,
                                      BagMinMaxScaler, BagNormalizer, BagPCA)


################################################################################

def test_basic():
    bags = [np.random.normal(5, 3, size=(np.random.randint(10, 100), 20))
            for _ in xrange(50)]
    feats = Features(bags, stack=True)

    stder = BagStandardizer()
    stdized = stder.fit_transform(bags)
    stdized.make_stacked()

    assert np.allclose(np.mean(stdized.stacked_features), 0)
    assert np.allclose(np.std(stdized.stacked_features), 1)

    first_five = stder.transform(bags[:5])
    assert first_five == stdized[:5]

    minmaxer = BagMinMaxScaler([3, 7])
    minmaxed = minmaxer.fit_transform(feats)
    minmaxed.make_stacked()
    assert np.allclose(np.min(minmaxed.stacked_features, 0), 3)
    assert np.allclose(np.max(minmaxed.stacked_features, 0), 7)

    normer = BagNormalizer('l1')
    normed = normer.fit_transform(Features(bags))
    normed.make_stacked()
    assert np.allclose(np.sum(np.abs(normed.stacked_features), 1), 1)

    class GetMean(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X.mean(axis=1)[None, :]
    m = BagPreprocesser(GetMean())
    assert_raises(ValueError, lambda: m.transform(bags))


def test_pca():
    bags = [np.random.normal(5, 3, size=(np.random.randint(10, 100), 20))
            for _ in xrange(50)]
    feats = Features(bags, stack=True)

    pca = BagPCA(k=3)
    pca.fit(bags)
    pcaed = pca.transform(bags)
    assert pcaed.dim == 3

    BagPCA(varfrac=.3).fit_transform(bags)

    pca2 = BagPCA(k=20)
    pcaed2 = pca2.fit_transform(bags)
    orig = pca2.inverse_transform(pcaed2)
    orig.make_stacked()
    assert np.allclose(feats.stacked_features, orig.stacked_features)

    assert BagPCA(k=5, randomize=True).fit_transform(bags).dim == 5

    assert_raises(TypeError, lambda: BagPCA(randomize=True))
    assert_raises(TypeError, lambda: BagPCA(mle_components=True, k=12))
    assert BagPCA(mle_components=True)



################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
