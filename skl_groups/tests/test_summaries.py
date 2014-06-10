from __future__ import division

from copy import copy, deepcopy
from functools import partial
import os
import sys
import warnings

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals.six.moves import xrange, cPickle as pickle
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups import Features
from skl_groups.summaries import BagOfWords


################################################################################

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
    assert bow.codewords.shape == (n_codewords, dim)
    assert np.all(bowed >= 0)
    assert np.all(np.sum(bowed, 1) == [b.shape[0] for b in bags])

    bow.fit(Features(bags))
    bowed2 = bow.transform(bags)
    assert np.all(bowed == bowed2)
    assert bow.codewords.shape == (n_codewords, dim)

    minikmeans = MiniBatchKMeans(n_clusters=n_codewords, max_iter=100,
                                 random_state=47)
    minibow = BagOfWords(minikmeans)
    assert_raises(AttributeError, lambda: minibow.transform(bags))
    minibowed = minibow.fit_transform(bags)
    assert minibowed.shape == bowed.shape
    assert np.all(bowed >= 0)
    assert np.all(np.sum(bowed, 1) == [b.shape[0] for b in bags])


################################################################################

if __name__ == '__main__':
    import nose
    nose.main()