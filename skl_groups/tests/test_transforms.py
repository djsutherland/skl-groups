from __future__ import division

import os
import sys

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from nose.tools import assert_raises

if __name__ == '__main__':
    # make this copy of skl_groups importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from skl_groups.kernels.transform import (
        PairwisePicker, Symmetrize, RBFize,
        ProjectPSD, FlipPSD, ShiftPSD, SquarePSD)

################################################################################

def test_pairwise_picker():
    X = np.arange(64).reshape((2, 2, 4, 4))
    pick = PairwisePicker((0, 1))
    Xt = pick.fit_transform(X)
    assert np.all(Xt == np.arange(16, 32).reshape((4, 4)))

    pick = PairwisePicker[:, 0]
    assert np.all(pick.fit_transform(X) == X[:, 0])


def test_symmetrize():
    X = np.random.normal(size=(10, 10))
    Xo = X.copy()

    sym = Symmetrize(copy=True)
    Xs = sym.fit_transform(X)
    assert np.all(X == Xo)
    assert np.allclose(Xs, (X + X.T) / 2)
    assert np.all(sym.transform(X + 1) == X + 1)

    Xs2 = Symmetrize(copy=False).fit_transform(X)
    assert np.allclose(Xs2, (X + X.T) / 2)

    sym = Symmetrize()
    assert_raises(NotImplementedError, lambda: sym.fit(X))
    assert_raises(TypeError, lambda: sym.fit_transform(np.zeros((5, 3))))


def test_rbfize():
    X = np.random.normal(size=(20, 4))
    dists = euclidean_distances(X)
    median = np.median(dists[np.triu_indices_from(dists, k=1)])

    rbf = RBFize(gamma=.25)
    res = rbf.fit_transform(dists)
    assert not hasattr(res, 'median_')
    assert np.allclose(res, np.exp(-.25 * dists ** 2))

    rbf = RBFize(gamma=.25, squared=True)
    res = rbf.fit_transform(dists)
    assert np.allclose(res, np.exp(-.25 * dists))

    rbf = RBFize(gamma=4, scale_by_median=True)
    res = rbf.fit_transform(dists)
    assert np.allclose(rbf.median_, median)
    assert np.allclose(res, np.exp((-4 * median**2) * dists ** 2))

    rbf = RBFize(gamma=4, scale_by_median=True, squared=True)
    res = rbf.fit_transform(dists)
    assert np.allclose(rbf.median_, median)
    assert np.allclose(res, np.exp((-4 * median) * dists))



def test_project():
    X = np.diagflat([-2, -1, 0, 1, 2])
    # eigenvalues -2, -1, 0, 1, 2; eigenvectors are I

    Xproj = ProjectPSD().fit_transform(X)
    assert np.allclose(Xproj, np.diagflat([0, 0, 0, 1, 2]))

    Xproj2 = ProjectPSD().fit(X).transform(X)
    assert np.allclose(Xproj2, np.diagflat([0, 0, 0, 1, 2]))

    Xproj3 = ProjectPSD(negatives_likely=True).fit(X).transform(X[:3, :])
    assert np.allclose(Xproj3, np.zeros((3, 5)))

    Xproj4 = ProjectPSD(negatives_likely=False).fit(X).transform(X[:3, :])
    assert np.allclose(Xproj4, np.zeros((3, 5)))

    Xproj5 = ProjectPSD(negatives_likely=True, copy=False, min_eig=.5) \
            .fit_transform(X.copy())
    assert np.allclose(Xproj5, np.diagflat([.5, .5, .5, 1, 2]))

    Xproj6 = ProjectPSD(negatives_likely=True, copy=False, min_eig=.5) \
            .fit(X.copy()).transform(X.copy())
    assert np.allclose(Xproj6, np.diagflat([.5, .5, 0, 1, 2]))

    assert_raises(TypeError, lambda: ProjectPSD().fit(X[:2, :]))
    assert_raises(TypeError, lambda: ProjectPSD().fit_transform(X[:2, :]))
    assert_raises(TypeError, lambda: ProjectPSD().fit(X).transform(X[:, :2]))


def test_flip():
    X = np.diagflat([-2, -1, 0, 1, 2])
    # eigenvalues -2, -1, 0, 1, 2; eigenvectors are I

    Xflip = FlipPSD().fit_transform(X)
    assert np.allclose(Xflip, np.diagflat([2, 1, 0, 1, 2]))

    Xflip2 = FlipPSD().fit(X).transform(X)
    assert np.allclose(Xflip2, np.diagflat([2, 1, 0, 1, 2]))

    Xflip3 = FlipPSD().fit(X).transform(X[:3, :])
    assert np.allclose(Xflip3, [[2,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0]])

    assert_raises(TypeError, lambda: FlipPSD().fit(X[:2, :]))
    assert_raises(TypeError, lambda: FlipPSD().fit_transform(X[:2, :]))
    assert_raises(TypeError, lambda: FlipPSD().fit(X).transform(X[:, :2]))


def test_shift():
    X = np.diagflat([-2., -1, 0, 1, 2])
    # eigenvalues -2, -1, 0, 1, 2; eigenvectors are I

    Xshift = ShiftPSD().fit_transform(X)
    assert np.allclose(Xshift, np.diagflat([0, 1, 2, 3, 4]))

    Xshift2 = ShiftPSD().fit(X).transform(X)
    assert np.allclose(Xshift2, np.diagflat([0, 1, 2, 3, 4]))

    Xshift3 = ShiftPSD().fit(X).transform(X[:3, :])
    assert np.allclose(Xshift3, X[:3, :])

    Xshift4 = ShiftPSD(min_eig=2).fit_transform(X)
    assert np.allclose(Xshift4, np.diagflat([2, 3, 4, 5, 6]))

    assert_raises(TypeError, lambda: ShiftPSD().fit(X[:2, :]))
    assert_raises(TypeError, lambda: ShiftPSD().fit_transform(X[:2, :]))
    assert_raises(TypeError, lambda: ShiftPSD().fit(X).transform(X[:, :2]))


def test_square():
    X = np.diagflat([-2., -1, 0, 1, 2])
    # eigenvalues -2, -1, 0, 1, 2; eigenvectors are I

    Xsquare = SquarePSD().fit_transform(X)
    assert np.allclose(Xsquare, np.diagflat([4, 1, 0, 1, 4]))

    assert_raises(NotImplementedError, lambda: SquarePSD().fit(X))
    assert_raises(NotImplementedError, lambda: SquarePSD().transform(X))
    assert_raises(TypeError, lambda: SquarePSD().fit_transform(X[:2, :]))


################################################################################

if __name__ == '__main__':
    import nose
    nose.main()
