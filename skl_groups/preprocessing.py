import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.utils import check_array

from .features import Features, as_features

# TODO: support inplace transformations here

class BagPreprocesser(BaseEstimator, TransformerMixin):
    '''
    Applies a preprocessing estimator to each column of the bags, independently.
    This is a reasonable thing to do for some cases
    (especially :class:`sklearn.preprocessing.StandardScaler`,
    :class:`sklearn.preprocessing.MinMaxScaler`,
    :class:`sklearn.decomposition.PCA`).
    It's not a reasonable thing to do for things that rely on interactions
    between points, or that change the number of output points. (Changing
    the dimension is okay.)

    Parameters
    ----------
    transformer : an sklearn transformer
        The transformer to apply to the stacked features. Must return the
        same number of features.
    '''
    def __init__(self, transformer):
        t = transformer
        if (not (hasattr(t, "fit") or hasattr(t, 'fit_transform')) or
                not hasattr(t, "transform")):
            raise TypeError("The transformer doesn't have appropriate methods.")
        self.transformer = t

    def _gather_outputs(self, old, new):
        if new.shape[0] != old.total_points:
            msg = "Transformer changed number of points from {} to {}"
            raise ValueError(msg.format(old.total_points, new.shape[0]))
        return Features(new, old.n_pts, **old.meta)

    def fit(self, X, y=None, **params):
        '''
        Fit the transformer on the stacked points.

        Parameters
        ----------
        X : :class:`Features` or list of arrays of shape ``[n_samples[i], n_features]``
            Training set. If a Features object, it will be stacked.

        any other keyword argument :
            Passed on as keyword arguments to the transformer's ``fit()``.
        '''
        X = as_features(X, stack=True)
        self.transformer.fit(X.stacked_features, y, **params)
        return self

    def transform(self, X, **params):
        '''
        Transform the stacked points.

        Parameters
        ----------
        X : :class:`Features` or list of bag feature arrays
            New data to transform.

        any other keyword argument :
            Passed on as keyword arguments to the transformer's ``transform()``.

        Returns
        -------
        X_new : :class:`Features`
            Transformed features.
        '''
        X = as_features(X, stack=True)
        X_new = self.transformer.transform(X.stacked_features, **params)
        return self._gather_outputs(X, X_new)

    def fit_transform(self, X, y=None, **params):
        '''
        Fit and transform the stacked points.

        Parameters
        ----------
        X : :class:`Features` or list of bag feature arrays
            Data to train on and transform.

        any other keyword argument :
            Passed on as keyword arguments to the transformer's ``transform()``.

        Returns
        -------
        X_new : :class:`Features`
            Transformed features.
        '''
        X = as_features(X, stack=True)
        X_new = self.transformer.fit_transform(X.stacked_features, y, **params)
        return self._gather_outputs(X, X_new)

    def inverse_transform(self, X, **params):
        '''
        Transform data back to its original space, i.e., return an input
        X_original whose transform would (maybe approximately) be X.

        Parameters
        ----------
        X : :class:`Features` or list of bag feature arrays
            Data to train on and transform.

        any other keyword argument :
            Passed on as keyword arguments to the transformer's 
            ``inverse_transform()``.

        Returns
        -------
        X_original : :class:`Features`
        '''
        X = as_features(X, stack=True)
        Xo = self.transformer.inverse_transform(X.stacked_features, **params)
        return self._gather_outputs(X, Xo)


class BagStandardizer(BagPreprocesser):
    '''
    Standardizes each feature dimension to have zero mean and unit variance,
    regardless of the bag it falls into.

    This is just :class:`BagPreprocesser` with
    :class:`sklearn.preprocessing.StandardScaler`.
    '''
    def __init__(self):
        super(BagStandardizer, self).__init__(StandardScaler())


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Standardizes features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The standardization is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This standardization is often used as an alternative to zero mean,
    unit variance scaling.

    Notes
    -----

    This is a version of :class:`sklearn.preprocessing.MinMaxScaler`
    with support for truncation added. It's been
    `proposed <https://github.com/scikit-learn/scikit-learn/pull/3342>`_
    for inclusion in scikit-learn, but is not yet in there.

    Parameters
    ----------
    feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : boolean, optional, default is True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    truncate : boolean, optional, default is False
        If True, :meth:`transform` will truncate any inputs that lie outside
        the min/max of the values passed to :meth:`fit` to lie on the ends
        of feature_range. Normally, the transform of these points will be
        outside feature_range.

    fit_feature_range : None or tuple (min, max), default None
        If not None, :meth:`fit` will actually rescale such that the passed
        features all lie within fit_feature_range rather than just
        feature_range. This is useful when truncate is True, to give
        some "wiggle room" before it starts truncating. Otherwise it just
        effectively overrides feature_range.

    Attributes
    ----------
    `min_` : ndarray, shape (n_features,)
        Per feature adjustment for minimum.

    `scale_` : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
    """

    def __init__(self, feature_range=(0, 1), copy=True, truncate=False,
                 fit_feature_range=None):
        self.feature_range = feature_range
        self.copy = copy
        self.truncate = truncate
        self.fit_feature_range = fit_feature_range

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy,
                        dtype=[np.float64, np.float32, np.float16, np.float128])

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        if self.fit_feature_range is not None:
            fit_feature_range = self.fit_feature_range
            if fit_feature_range[0] >= fit_feature_range[1]:
                raise ValueError("Minimum of desired (fit) feature range must "
                                 "be smaller than maximum. Got %s."
                                 % str(feature_range))
            if (fit_feature_range[0] < feature_range[0] or
                    fit_feature_range[1] > feature_range[1]):
                raise ValueError("fit_feature_range must be a subset of "
                                 "feature_range. Got %s, fit %s."
                                 % (str(feature_range),
                                    str(fit_feature_range)))
            feature_range = fit_feature_range

        data_min = np.min(X, axis=0)
        data_range = np.max(X, axis=0) - data_min
        # Do not scale constant features
        data_range[data_range == 0.0] = 1.0
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range
        self.data_min = data_min
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        X = check_array(X, copy=self.copy)
        X *= self.scale_
        X += self.min_
        if self.truncate:
            np.maximum(self.feature_range[0], X, out=X)
            np.minimum(self.feature_range[1], X, out=X)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Note that if truncate is true, any truncated points will not
        be restored exactly.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        X = check_array(X, copy=self.copy)
        X -= self.min_
        X /= self.scale_
        return X



class BagMinMaxScaler(BagPreprocesser):
    '''
    Linearly scales each feature dimension to lie within the given range, for
    example [0, 1].

    This is just :class:`BagPreprocesser` with :class:`MinMaxScaler`.

    Parameters
    ----------
    feature_range : tuple (min, max), default = (0, 1)
        Desired range of the transformed data.
    '''
    def __init__(self, feature_range=(0, 1), truncate=False,
                 fit_feature_range=None):
        super(BagMinMaxScaler, self).__init__(MinMaxScaler(
            feature_range=feature_range, truncate=truncate,
            fit_feature_range=fit_feature_range))


class BagNormalizer(BagPreprocesser):
    '''
    Normalizes each sample individually to have unit norm (l1 or l2).

    This is just :class:`BagPreprocesser` with
    :class:`sklearn.preprocessing.Normalizer`.

    Parameters
    ----------
    norm : 'l1' or 'l2', optional, default 'l2'
        The norm to use to normalize each nonzero sample.
    '''
    def __init__(self, norm='l2'):
        super(BagNormalizer, self).__init__(Normalizer(norm))


DEFAULT_VARFRAC = 0.7
class BagPCA(BagPreprocesser):
    '''
    Runs principal components analysis to reduce the dimensionality of the
    features.

    This is just :class:`BagPreprocesser` with
    either :class:`sklearn.decomposition.PCA`
    or :class:`sklearn.decomposition.RandomizedPCA`.

    Parameters
    ----------
    k : int, optional
        The dimensionality to reduce to.

    mle_components : boolean, optional, default False
        Use Minka's MLE for determining the number of components.

    varfrac : float in (0, 1], optional, default 0.7
        Use enough components to cover this fraction of the variance.
        Only one of {k, mle_components, varfrac} can be passed.

    randomize : boolean, optional, default False
        Use a randomized PCA. This can be faster and less memory-intensive for
        large inputs, but is approximate and requires specifying an explicit
        number of components.

    whiten : boolean, optional, default False
        Whether to whiten the outputs, by dividing the components by the
        singular values. This removes some information, but makes the variance
        of the outputs the identity matrix.
    '''
    def __init__(self, k=None, mle_components=False, varfrac=None,
                 randomize=False, whiten=False):
        n_specs = sum(1 for x in [k, mle_components, varfrac] if x)
        if n_specs > 1:
            msg = "can't specify number of components in more than one way"
            raise TypeError(msg)
        if n_specs == 0:
            varfrac = DEFAULT_VARFRAC

        if randomize:
            if k is None:
                raise TypeError("can't do random PCA without a specific k")
            pca = RandomizedPCA(k, whiten=whiten)
        else:
            if k is not None:
                n_components = k
            elif mle_components:
                n_components = 'mle'
            elif varfrac is not None:
                n_components = varfrac
            pca = PCA(n_components, whiten=whiten)
        super(BagPCA, self).__init__(pca)
