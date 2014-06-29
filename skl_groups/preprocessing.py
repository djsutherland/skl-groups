from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from .features import Features

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

    def _check_inputs(self, X):
        if isinstance(X, Features):
            X.make_stacked()
        else:
            X = Features(X, stack=True)
        return X

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
        X = self._check_inputs(X)
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
        X = self._check_inputs(X)
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
        X = self._check_inputs(X)
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
        X = self._check_inputs(X)
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


class BagMinMaxScaler(BagPreprocesser):
    '''
    Linearly scales each feature dimension to lie within the given range, for
    example [0, 1].

    This is just :class:`BagPreprocesser` with
    :class:`sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    feature_range : tuple (min, max), default = (0, 1)
        Desired range of the transformed data.
    '''
    def __init__(self, feature_range=(0, 1)):
        super(BagMinMaxScaler, self).__init__(MinMaxScaler(feature_range))


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
