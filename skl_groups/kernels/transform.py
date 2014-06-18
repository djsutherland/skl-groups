import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.six import with_metaclass
from sklearn.externals.six.moves import xrange


class _Meta(type):
    def __getitem__(cls, indices):
        return cls(indices)

class PairwisePicker(BaseEstimator, TransformerMixin, with_metaclass(_Meta)):
    '''
    Picks a subset of the passed-in matrix. Useful for choosing one of several
    divergences computed by :class:`sklearn.divergences.KNNDivergenceEstimator`,
    for example.

    Rather than ``PairwisePicker((0, slice(2, 3)))``, you can also do
    ``PairwisePicker[0, 2:3]``.

    Parameters
    ----------
    indices : tuple of integers / slice objects / etc
        The indices to subset the input with.
    '''
    _pairwise = True  # TODO: not really
    _pairwise_output = True

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X=None, y=None):
        "Do nothing; this transformer is stateless."
        return self

    def transform(self, X):
        "Subsets the given matrix."
        return X[self.indices]


class Symmetrize(BaseEstimator, TransformerMixin):
    '''
    Symmetrizes pairwise affinity/distance/whatever matrices, by taking
    the mean of itself and its transpose.

    Parameters
    ----------
    copy : boolean, optional, default True
        If false, invalidates the passed-in matrix.

    Notes
    -----
    ``copy=False`` currently doesn't do anything.
    '''

    _pairwise = True
    _pairwise_output = True

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        "Raises NotImplementedError."
        raise NotImplementedError("Symmetrize can only fit_transform")

    def transform(self, X):
        "Raises NotImplementedError."
        raise NotImplementedError("Symmetrize can only fit_transform")

    def fit_transform(self, X, y=None):
        '''
        Symmetrizes X.

        Parameters
        ----------
        X : array, shape [n, n]
            The pairwise inputs to symmetrize.

        Returns
        -------
        X : array, shape [n, n]
            The symmetrized pairwise outputs.
        '''
        # TODO: figure out a no-copy version of this...
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        X = X + X.T
        X /= 2
        return X


class ProjectPSD(BaseEstimator, TransformerMixin):
    '''
    Projects a pairwise square symmetric affinity matrix to be positive
    semidefinite, by discarding any negative eigenvalues from its spectrum.

    `fit_transform()` does the actual projection. If you `transform` onto data
    different than the data originally `fit` on (not necessarily square), then
    the data will be processed in a way that attempts to treat test similarities
    consistently with training ones, using the method of [1].

    Parameters
    ----------
    min_eig : float, optional, default 0
        The minimum eigenvalue for the projected matrix. Because of
        floating-point inaccuracies, don't take this too literally.

    copy : boolean, optional, default True
        Operate on a copy of the passed-in matrix; otherwise, the original
        matrix will be invalidated.

    negatives_likely : boolean, optional, default True
        Optimize memory usage for the case where we expect there to be negative
        eigenvalues.

    Attributes
    ----------
    `clip_` : array of shape (n, n)
        The linear transformation corresponding to
        the clip operation on the training points.

    References
    ----------
    [1] Y. Chen, E. K. Garcia, M. R. Gupta, A. Rahimi, & L. Cazzanti (2009).
        Similarity-based classification: Concepts and algorithms.
        Journal of Machine Learning Research, 10, 747-776.
    '''

    _pairwise = True
    _pairwise_output = True

    def __init__(self, min_eig=0, copy=True, negatives_likely=True):
        self.min_eig = min_eig
        self.copy = copy
        self.negatives_likely = negatives_likely

    def fit(self, X, y=None):
        '''
        Learn the linear transformation to clipped eigenvalues.

        Note that if min_eig isn't zero and any of the original eigenvalues
        were exactly zero, this will leave those eigenvalues as zero.

        Parameters
        ----------
        X : array, shape [n, n]
            The *symmetric* input similarities. If X is asymmetric, it will be
            treated as if it were symmetric based on its lower-triangular part.
        '''
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        # TODO: only get negative eigs somehow?
        vals, vecs = scipy.linalg.eigh(X, overwrite_a=not self.copy)
        vals = vals.reshape(-1, 1)

        if self.min_eig == 0:
            inner = vals > self.min_eig
        else:
            with np.errstate(divide='ignore'):
                inner = np.where(vals >= self.min_eig, 1,
                                 np.where(vals == 0, 0, self.min_eig / vals))

        self.clip_ = np.dot(vecs, inner * vecs.T)
        return self

    def transform(self, X):
        '''
        Transforms X according to the linear transformation corresponding to
        clipping the input eigenvalues.

        Parameters
        ----------
        X : array, shape [n_test, n]
            The test similarities to training points.

        Returns
        -------
        Xt : array, shape [n_test, n]
            The transformed test similarites to training points.
        '''
        n = self.clip_.shape[0]
        if X.ndim != 2 or X.shape[1] != n:
            msg = "X should have {} columns, the number of samples at fit time"
            raise TypeError(msg.format(self.clip_.shape[0]))
        return np.dot(X, self.clip_)

    def fit_transform(self, X, y=None):
        '''
        Clips the negative eigenvalues of X.

        Parameters
        ----------
        X : array, shape [n, n]
            The *symmetric* input similarities. If X is asymmetric, it will be
            treated as if it were symmetric based on its lower-triangular part.

        Returns
        -------
        Xt : array, shape [n, n]
            The transformed training similarities; smallest eigenvalue will be
            at least `self.min_eig`.
        '''
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        discard_X = not self.copy and self.negatives_likely
        vals, vecs = scipy.linalg.eigh(X, overwrite_a=discard_X)
        vals = vals[:, None]

        self.clip_ = np.dot(vecs, (vals > self.min_eig) * vecs.T)

        if discard_X or vals[0, 0] < self.min_eig:
            del X
            np.maximum(vals, self.min_eig, out=vals)
            X = np.dot(vecs, vals * vecs.T)
            del vals, vecs

            # should be symmetric, but make sure because floats
            X = Symmetrize(copy=False).fit_transform(X)
        return X


class FlipPSD(BaseEstimator, TransformerMixin):
    '''
    Makes a pairwise symmetric square affinitimy matrix into a valid positive
    semidefinite kernel, by flipping the sign of any negative eigenvalues in
    its spectrum.

    `fit_transform()` does the actual projection. If you `transform` onto data
    different than the data originally `fit` on (not necessarily square), then
    the data will be processed in a way that attempts to treat test similarities
    consistently with training ones, using the method of [1].

    Parameters
    ----------
    copy : boolean, optional, default True
        Operate on a copy of the passed-in matrix; otherwise, the original
        matrix will be invalidated (for both ``fit()`` and ``transform()``).

    negatives_likely : boolean, optional, default True
        Optimize memory usage for the case where we expect there to be negative
        eigenvalues.

    Attributes
    ----------
    `flip_` : array of shape (n, n)
        The linear transformation corresponding
        to the flip operation on the training points.

    References
    ----------
    [1] Y. Chen, E. K. Garcia, M. R. Gupta, A. Rahimi, & L. Cazzanti (2009).
        Similarity-based classification: Concepts and algorithms.
        Journal of Machine Learning Research, 10, 747-776.
    '''

    _pairwise = True
    _pairwise_output = True

    def __init__(self, copy=True, negatives_likely=True):
        self.copy = copy
        self.negatives_likely = negatives_likely

    def fit(self, X, y=None):
        '''
        Learn the linear transformation to flipped eigenvalues.

        Parameters
        ----------
        X : array, shape [n, n]
            The *symmetric* input similarities. If X is asymmetric, it will be
            treated as if it were symmetric based on its lower-triangular part.
        '''
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        # TODO: only get negative eigs somehow?
        vals, vecs = scipy.linalg.eigh(X, overwrite_a=not self.copy)
        vals = vals[:, None]

        self.flip_ = np.dot(vecs, np.sign(vals) * vecs.T)
        return self

    def transform(self, X):
        '''
        Transforms X according to the linear transformation corresponding to
        flipping the input eigenvalues.

        Parameters
        ----------
        X : array, shape [n_test, n]
            The test similarities to training points.

        Returns
        -------
        Xt : array, shape [n_test, n]
            The transformed test similarites to training points.
        '''
        n = self.flip_.shape[0]
        if X.ndim != 2 or X.shape[1] != n:
            msg = "X should have {} columns, the number of samples at fit time"
            raise TypeError(msg.format(self.flip_.shape[0]))
        return np.dot(X, self.flip_)

    def fit_transform(self, X, y=None):
        '''
        Flips the negative eigenvalues of X.

        Parameters
        ----------
        X : array, shape [n, n]
            The *symmetric* input similarities. If X is asymmetric, it will be
            treated as if it were symmetric based on its lower-triangular part.

        Returns
        -------
        Xt : array, shape [n, n]
            The transformed training similarities.
        '''
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        discard_X = not self.copy and self.negatives_likely
        vals, vecs = scipy.linalg.eigh(X, overwrite_a=discard_X)
        vals = vals[:, None]

        self.clip_ = np.dot(vecs, np.sign(vals) * vecs.T)

        if discard_X or vals[0, 0] < 0:
            del X
            np.abs(vals, out=vals)
            X = np.dot(vecs, vals * vecs.T)
            del vals, vecs

            # should be symmetric, but make sure because floats
            X = Symmetrize(copy=False).fit_transform(X)
        return X


class ShiftPSD(BaseEstimator, TransformerMixin):
    '''
    Makes a pairwise squar affinitimy matrix into a valid positive semidefinite
    kernel, by flipping the sign of any negative eigenvalues in its spectrum.

    `fit_transform()` does the actual projection. If you `transform` onto data
    different than the data originally `fit` on (not necessarily square), then
    nothing will be changed, because the shift operation only affects self-
    similarities.

    Parameters
    ----------
    min_eig : float, optional, default 0
        The minimum eigenvalue for the projected matrix. Because of
        floating-point inaccuracies, don't take this too literally.

    copy : boolean, optional, default True
        Operate on a copy of the passed-in matrix; otherwise, the original
        matrix will be invalidated (for both ``fit()`` and ``transform()``).

    Attributes
    ----------
    `train_` : array of shape [n, n]
        The training similarities.
        Stored so that `transform` can check if
        it's transforming the test data and act appropriately.

    `shift_` : float
        The amount to shift all the eigenvalues up by.
    '''

    _pairwise = True
    _pairwise_output = True

    def __init__(self, min_eig=0, copy=True):
        self.min_eig = min_eig
        self.copy = copy

    def fit(self, X, y=None):
        '''
        Learn the transformation to shifted eigenvalues. Only depends
        on the input dimension.

        Parameters
        ----------
        X : array, shape [n, n]
            The *symmetric* input similarities.
        '''
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        self.train_ = X
        lo, = scipy.linalg.eigvalsh(X, eigvals=(0, 0))
        self.shift_ = max(self.min_eig - lo, 0)
        return self

    def transform(self, X):
        '''
        Transforms X according to the linear transformation corresponding to
        shifting the input eigenvalues to all be at least ``self.min_eig``.

        Parameters
        ----------
        X : array, shape [n_test, n]
            The test similarities to training points.

        Returns
        -------
        Xt : array, shape [n_test, n]
            The transformed test similarites to training points. Only different
            from X if X is the training data.
        '''
        n = self.train_.shape[0]
        if X.ndim != 2 or X.shape[1] != n:
            msg = "X should have {} columns, the number of samples at fit time"
            raise TypeError(msg.format(n))

        if self.copy:
            X = X.copy()

        if self.shift_ != 0 and X is self.train_ or np.all(X == self.train_):
            X[xrange(n), xrange(n)] += self.shift_
        return X


class SquarePSD(BaseEstimator, TransformerMixin):
    '''
    Makes a pairwise square affinity matrix into a valid positive semidefinite
    kernel by symmetrizing then squaring its eigenvalues (via S -> S S^T).
    Equivalent to using the similarities to training points as features in a
    linear classifier.

    Parameters
    ----------
    copy : boolean, optional, default True
        Operate on a copy of the passed-in matrix; otherwise, the original
        matrix will be invalidated.
    '''

    _pairwise = True
    _pairwise_output = True

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        raise NotImplementedError("SquarePSD can only fit_transform().")

    def transform(self, X):
        raise NotImplementedError("SquarePSD can only fit_transform().")

    def fit_transform(self, X, y=None):
        n = X.shape[0]
        if X.shape != (n, n):
            raise TypeError("Input must be a square matrix.")

        return X.dot(X.T)
