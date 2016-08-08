from __future__ import division

from copy import copy, deepcopy
from functools import partial
import os
import sys
import warnings

import numpy as np
from sklearn.externals.six.moves import xrange, cPickle as pickle
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.features import Features


wrong_type = partial(assert_raises, (TypeError, IndexError))


################################################################################

def test_features_basic():
    bags = [np.random.normal(size=(np.random.randint(10, 100), 10))
            for _ in xrange(20)]

    assert repr(Features([[[8, 9], [12, 12]]]))

    feats = Features(bags, copy=False, stack=False, label=np.arange(20))
    assert len(feats) == 20
    assert feats.total_points == sum(bag.shape[0] for bag in bags)
    assert np.all(feats[3] == bags[3])
    assert np.all(feats.label == np.arange(20))
    assert repr(feats)
    assert feats.dtype == np.float64
    assert feats != bags
    assert feats.bare() == bags

    sub = feats[[5, 2]]
    assert np.all(sub.label == [5, 2])
    assert np.all(sub[0] == feats[5])
    assert np.all(sub[1] == feats[2])
    assert repr(sub)

    feats[4][0, 0] = 1000
    assert bags[4][0, 0] == 1000

    feats.make_stacked()
    assert feats != bags
    assert feats.bare() == bags
    assert len(feats) == 20
    assert feats.total_points == sum(bag.shape[0] for bag in bags)
    assert np.all(feats[3] == bags[3])
    assert np.all(feats.label == np.arange(20))
    feats[0][0, 0] = -800
    assert feats.features[0][0, 0] == -800
    assert feats.stacked_features[0, 0] == -800
    assert repr(feats)

    wrong_type(lambda: feats['label'])
    wrong_type(lambda: feats[['label']])
    wrong_type(lambda: feats[[3, 'label']])

    cop = feats.copy()
    assert cop == feats
    assert cop.stacked
    cop.make_stacked()
    assert cop == feats
    cop[0][0, 0] = 12
    assert cop != feats
    assert repr(cop)

    fs = lambda *a, **kw: partial(Features, *a, **kw)

    bags = np.random.normal(size=(10, 5))
    wrong_type(fs(bags))
    wrong_type(fs(bags, [[4], [12]]))
    wrong_type(fs(bags, []))
    wrong_type(fs(bags, [-3, 13]))
    wrong_type(fs(bags, [7.5, 2.5]))
    wrong_type(fs(bags, [7, 2]))
    wrong_type(fs(np.zeros((10, 0)), [7, 3]))

    bags = [np.random.normal(size=(5, 8)), np.random.normal(size=(6, 8))]
    wrong_type(fs(bags, [5, 6]))
    assert np.all(
        Features([[5, 6], [[7, 9], [0, 0]]])[0] == np.reshape([5, 6], (1, 2)))
    wrong_type(fs([ [[[5]]] ]))
    wrong_type(fs([["hello", "there"]]))
    wrong_type(fs([[np.arange(10, dtype=int)], [np.arange(10, dtype=float)]]))
    wrong_type(fs([np.random.randn(8, 7), np.random.randn(0, 7)]))

    assert np.all(
        Features([[[1, 2]], [[3, 4]]], stack=True).stacked_features
        == [[1, 2], [3, 4]])

    assert_raises(ValueError, fs(bags, labels=np.arange(3)))

    with warnings.catch_warnings(record=True) as w:
        Features(bags, total_points=[1, 2])
        assert len(w) == 1


def test_copy_constructor():
    bags = [np.random.normal(size=(np.random.randint(10, 100), 10))
            for _ in xrange(20)]
    unstacked = Features(bags, label=np.arange(20))

    oth_unstacked = Features(unstacked)
    assert oth_unstacked.label is unstacked.label
    assert oth_unstacked.features[0] is unstacked.features[0]
    assert oth_unstacked == unstacked

    oth_unstacked_bare = Features(unstacked, bare=True)
    assert oth_unstacked_bare == bags
    assert oth_unstacked_bare.bare() == oth_unstacked_bare

    oth_unstacked = Features(unstacked, label=np.ones(20))
    assert np.all(oth_unstacked.label == 1)

    oth_unstacked2 = Features(unstacked, bare=True, label=np.arange(20))
    assert oth_unstacked2 == unstacked

    oth_unstacked_copy = Features(unstacked, copy=True)
    assert oth_unstacked_copy == unstacked
    assert not np.may_share_memory(oth_unstacked_copy.features[0],
                                   unstacked.features[0])

    stacked = unstacked.copy()
    stacked.make_stacked()
    oth_stacked = Features(stacked)
    assert oth_stacked == stacked


def test_copying():
    bags = [np.random.normal(size=(np.random.randint(10, 100), 10))
            for _ in xrange(20)]

    unstacked = Features(bags, copy=False, stack=False, label=np.arange(20))
    stacked = Features(bags, stack=True, label=np.arange(20))

    assert unstacked == stacked

    unstacked_copy = copy(unstacked)
    assert not unstacked_copy.stacked
    assert stacked == unstacked_copy == unstacked
    assert unstacked_copy.label is unstacked.label
    assert not np.may_share_memory(unstacked[0], unstacked_copy[0])

    unstacked_deep = deepcopy(unstacked)
    assert not unstacked_deep.stacked
    assert stacked == unstacked_deep == unstacked
    assert unstacked_deep.label is not unstacked.label

    stacked_copy = copy(stacked)
    assert stacked_copy.stacked
    assert stacked == stacked_copy == unstacked
    assert stacked_copy.label is stacked.label

    stacked_deep = deepcopy(stacked)
    assert stacked_deep.stacked
    assert stacked == stacked_deep == unstacked
    assert stacked_deep.label is not stacked.label

    unstacked_stacked = unstacked.copy(stack=True)
    assert unstacked_stacked.stacked
    assert stacked == unstacked_stacked == stacked
    assert unstacked_stacked.label is unstacked.label

    unstacked_pickled = pickle.loads(pickle.dumps(unstacked))
    assert unstacked == unstacked_pickled
    assert not unstacked_pickled.stacked
    assert unstacked_pickled.label is not unstacked.label

    stacked_pickled = pickle.loads(pickle.dumps(stacked))
    assert stacked == stacked_pickled
    assert stacked_pickled.stacked
    assert stacked_pickled.label is not stacked.label


def test_feats_add():
    bags = [np.random.normal(size=(np.random.randint(10, 100), 10))
            for _ in xrange(20)]
    labels = np.arange(20)

    first_15 = Features(bags[:15], labels=labels[:15])
    last_5 = Features(bags[15:], labels=labels[15:])

    plus = first_15 + last_5
    assert len(plus) == 20
    assert plus[:15] == first_15

    plus_list = first_15 + bags[15:]
    assert len(plus_list) == 20
    assert not plus_list.meta
    assert np.all(plus_list[16] == bags[16])

    plus_singlelist = first_15 + [bags[18]]
    assert np.all(plus_singlelist[15] == bags[18])

    rplus_list = bags[15:] + first_15
    assert np.all(rplus_list[0] == bags[15])

    rplus_singlelist = [bags[15]] + first_15
    assert np.all(rplus_singlelist[0] == bags[15])
    assert rplus_singlelist[1:] == first_15.bare()

    wrong_type(lambda: first_15 + 12)
    wrong_type(lambda: 12 + first_15)

    assert_raises(ValueError, lambda: first_15 + np.asarray(bags))

################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
