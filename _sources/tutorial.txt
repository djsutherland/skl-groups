Quick tutorial
==============

.. currentmodule:: skl_groups

Motivation
----------

Let's say we have a bunch of labeled data.
Given a new set of data, we want to be able to predict its label.
This is the standard machine learning problem setting of
classification (if the labels are distinct classes)
or regression (if the labels are continuous values).

Most algorithms to solve these problems operate by
extracting feature vectors from the underlying data,
and then finding regions in feature space that correspond to the classes,
or smooth-ish functions in that space that match the regression problem.

For example, if you're classifying emails as spam or not spam,
you might choose one feature dimension that
corresponds to the number of times that the word "account" is used,
another for the number of times "pharmacy" appears,
another for whether the sender is in your address book,
and so on.

These choices are very important:
even the fanciest learning algorithm can't do anything with the wrong features,
and even the simplest can often do very well with the right ones.

But sometime it's not easy to summarize your data in a single, fixed-length vector.
One particular case where it's difficult: if your data is really comprised of a set of points.

For example, consider the following data,
where the '+' and '-' indicate training labels
and the '?' is a test point:


.. plot::
    :context:

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # generate data
    np.random.seed(2)
    n = 3

    mvn = np.random.multivariate_normal
    def nudge(k):
        nudge = np.zeros((k, 2))
        nudge[:k//2, 0] = -.5
        nudge[k//2:, 0] = .5
        return nudge
    plus = [mvn([0, 0], [[.1**2,0],[0,.35**2]], size=k) + nudge(k)
            for k in np.random.randint(8, 20, size=n)]
    minus = [mvn([0, 0], [[.2**2,0],[0,.2**2]], size=k)
            for k in np.random.randint(8, 20, size=n)]
    test_pt = mvn([0, 0], [[.2**2,0],[0,.2**2]], size=np.random.randint(5, 15))


    # plot
    from scipy.stats import gaussian_kde

    left = -1.1
    right = 1.1
    top = 1.3
    bot = -1.3
    grid = np.mgrid[left:right:.01, bot:top+.01:.01]

    def plot(X, ax):
        v = gaussian_kde(X.T, .6).evaluate(grid.reshape(2, -1)).reshape(grid.shape[1:])
        v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
        ax.contour(grid[0], grid[1], v, [.12], cmap='hot')
        ax.scatter(*X.T, s=50, color='k')
        ax.axis('off')
        ax.set_xlim(left, right)
        ax.set_ylim(bot, top)

    plt.figure()
    shape = (4, n + 1)
    for i, X in enumerate(plus):
        ax = plt.subplot2grid(shape, (0, i), rowspan=2)
        # plt.subplot(2, n, i)
        plot(X, ax)
        plt.annotate('+', (.5, .15), xycoords='axes fraction',
                     fontsize='large', fontweight='extra bold')

    for i, X in enumerate(minus):
        ax = plt.subplot2grid(shape, (2, i), rowspan=2)
        # plt.subplot(2, n, n + i)
        plot(X, ax)
        plt.annotate('-', (.5, .2), xycoords='axes fraction',
                     fontsize='large', fontweight='extra bold')

    ax = plt.subplot2grid(shape, (1, n), rowspan=2)
    plot(test_pt, ax)
    plt.annotate('?', (.5, .2), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')


    # for later
    bags = plus + minus + [test_pt]
    labels = np.r_[np.ones(n, dtype=bool), np.zeros(n, dtype=bool), False]

    from skl_groups.features import Features
    feats = Features(bags, labels=labels)


The test set obviously belongs to the "-" class.
How can we get that answer with machine learning?


Feature representation
----------------------

First, how do we even represent the data?

In scikit-learn, where objects are represented by feature vectors,
most learning methods take in a 2d array
(:class:`numpy.ndarray` or sometimes :mod:`scipy.sparse`)
of shape ``(n, n_features)``
where each of the ``n`` rows represents a different vector.

In skl-groups, each object is instead a set of vectors.
We represent that as a Python :class:`list` of arrays,
where each array is of shape ``(n_pts[i], n_features)``.
The feature dimensionality must be consistent between different sets,
but the number of points in each set can vary (as it does above).
(This is why we use a list; numpy arrays don't support "ragged" arrays like this.)

Internally, most methods convert these lists into
:class:`features.Features` objects,
which provide a convenient way to access the data consistently,
and can optionally store any metadata you like along with each bag.

For example, here's how we made the data for the plot above::

    import numpy as np
    mvn = np.random.multivariate_normal

    np.random.seed(87)
    n = 3

    def nudge(k):
        nudge = np.zeros((k, 2))
        nudge[:k//2, 0] = -.5
        nudge[k//2:, 0] = .5
        return nudge
    plus = [mvn([0, 0], [[.1**2,0],[0,.35**2]], size=k) + nudge(k)
            for k in np.random.randint(8, 15, size=n)]

    minus = [mvn([0, 0], [[.2**2,0],[0,.2**2]], size=k)
            for k in np.random.randint(5, 15, size=n)]

    test_pt = mvn([0, 0], [[.2**2,0],[0,.2**2]], size=np.random.randint(5, 15))

    bags = plus + minus + [test_pt]
    labels = np.r_[np.ones(n, dtype=bool), np.zeros(n, dtype=bool), False]


We can then pass ``bags`` to skl-groups's learning methods.
Alternatively, we can make a :class:`features.Features` object::

    >>> from skl_groups.features import Features
    >>> feats = Features(bags, labels=labels)
    >>> feats
    <Features: 7 bags with 11 to 14 2-dimensional points (88 total)>
    >>> feats.labels
    array([ True,  True,  True, False, False, False, False], dtype=bool)
     
By passing ``labels``, now if we slice ``features``,
``labels`` will be kept track of too::

    >>> feats[:4]
    <Features: 4 bags with 11 to 14 2-dimensional points (50 total)>
    >>> feats[:4].labels
    array([ True,  True,  True, False], dtype=bool)

It's just a convenience, though; learning methods won't use it to "cheat."


Means
-----

Perhaps the most obvious way to summarize a set of vectors is their mean,
which we can get with :class:`summaries.BagMean`.
BagMean will turn a list of features into a single array,
which we can then plug into our favorite :ref:`sklearn classifier <sklearn:supervised-learning>` (or regressor, clusterer, whatever)::

    >>> from skl_groups.summaries import BagMean
    >>> BagMean().fit_transform(feats).round(2)
    array([[-0.02,  0.02],
           [-0.02,  0.04],
           [ 0.01,  0.02],
           [ 0.04,  0.05],
           [ 0.11,  0.07],
           [-0.01,  0.02],
           [ 0.02, -0.02]])

.. todo::
    Make this "live" like the plots are

Plotting the results:

.. plot::
    :context:

    plt.close('all')

    from skl_groups.summaries import BagMean
    means = BagMean().fit_transform(feats)

    plt.figure()
    for i in range(n):
        plt.annotate('+', means[i], fontsize='large', fontweight='bold', color='blue')
    for i in range(n, 2 * n):
        plt.annotate('-', means[i], fontsize='large', fontweight='bold', color='red')
    plt.annotate('?', means[-1], fontsize='large', fontweight='bold', color='cyan')
    plt.xlim(-.5, .5)
    plt.ylim(-.5, .5)
    # plt.axis('off')


In our toy example,
the mean isn't really enough to distinguish between the two classes.
(In fact, if you look at the distributions the points are drawn from,
all of them have expectation (0, 0).)

In this particular case,
the covariance matrix "flattened" into a vector would probably be good enough.
That's not generally useful enough to have an implementation in skl_groups,
but it'd be easy enough to implement yourself by tweaking the source of BagMean.


Bag of Words
------------

Maybe the next-most-popular way to summarize a set of features is through
"bag of words."

In natural language processing,
a common way to get a basic understanding of document is to simply
take the counts of each word used in a document, and stack those into a vector
(see :ref:`the sklearn docs <sklearn:text_feature_extraction>`).
Computer vision researchers perform an analogous process with arbitrary feature vectors:
"quantize" the inputs into a set of *codewords*,
i.e. points where you can reasonably replace each of the input vectors with a
codeword without losing too much information,
and then represent each set by the count of each codeword.

We can perform this in skl_groups with the :class:`summaries.BagOfWords` estimator,
which relies on k-means :ref:`clustering <sklearn:clustering>` to choose the codewords.
Here we use :class:`sklearn.cluster.KMeans` to do it::

    >>> from sklearn.cluster import KMeans
    >>> from skl_groups.summaries import BagOfWords
    >>> bow = BagOfWords(KMeans(n_clusters=6, max_iter=100, n_init=2))
    >>> bowized = bow.fit_transform(feats)
    >>> bowized
    array([[0, 3, 4, 2, 0, 5],
           [1, 4, 2, 3, 0, 3],
           [0, 3, 2, 2, 0, 4],
           [4, 0, 1, 0, 6, 1],
           [4, 0, 1, 1, 4, 2],
           [6, 0, 3, 0, 4, 1],
           [9, 0, 0, 0, 2, 1]], dtype=int32)

Plotting how the bags lie on the selected codewords
(codewords in red, the areas that map to them in pastel):

.. plot::
    :context:
    
    plt.close('all')

    from sklearn.cluster import KMeans
    from skl_groups.summaries import BagOfWords
    bow = BagOfWords(KMeans(n_clusters=6, max_iter=100, n_init=2))
    bowized = bow.fit_transform(feats)

    from scipy.spatial import Voronoi
    from colorized_voronoi import voronoi_finite_polygons_2d

    vor = Voronoi(bow.codewords_)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    vor_cmap = mpl.cm.Pastel1

    def plot(X, ax):
        # draw voronoi polygons
        for i, region in enumerate(regions):
            plt.fill(*zip(*vertices[region]),
                     color=vor_cmap(i / float(len(regions))), zorder=0)

        # white out the region outside the "extent of the shape"
        v = gaussian_kde(X.T, .6).evaluate(grid.reshape(2, -1)).reshape(grid.shape[1:])
        plt.contourf(grid[0], grid[1], v, [0, .12, 10],
                     colors=[(1, 1, 1, 1), (0, 0, 0, 0)], zorder=1)

        ax.scatter(*bow.codewords_.T, s=30, color='r')
        ax.scatter(*X.T, s=20, color='k')
        ax.axis('off')
        ax.set_xlim(left, right)
        ax.set_ylim(bot, top)

    plt.figure()
    shape = (4, n + 1)
    for i, X in enumerate(plus):
        ax = plt.subplot2grid(shape, (0, i), rowspan=2)
        plot(X, ax)
        plt.annotate('+', (.5, .15), xycoords='axes fraction',
                     fontsize='large', fontweight='extra bold')

    for i, X in enumerate(minus):
        ax = plt.subplot2grid(shape, (2, i), rowspan=2)
        plot(X, ax)
        plt.annotate('-', (.5, .2), xycoords='axes fraction',
                     fontsize='large', fontweight='extra bold')

    ax = plt.subplot2grid(shape, (1, n), rowspan=2)
    plot(test_pt, ax)
    plt.annotate('?', (.5, .2), xycoords='axes fraction',
                 fontsize='large', fontweight='extra bold')


These are much easier to distinguish
(the negatives mostly lie in the central two codewords, the positives mostly outside).
And we can solve our prediction problem::

    >>> from sklearn.svm import SVC
    >>> SVC(probability=True).fit(bowized[:6], labels[:6]).predict(bowized[6:])
    array([False], dtype=bool)

The transformed features are six-dimensional, so we can't just plot them directly.
But we can plot a low-dimensional embedding where distances between the points
approximately correspond to distances in the high-dimensional space with
:class:`sklearn.manifold.LocallyLinearEmbedding`:

.. plot::
    :context:

    plt.close('all')

    from sklearn.manifold import LocallyLinearEmbedding

    lle = LocallyLinearEmbedding(n_components=2)
    lowdim = lle.fit_transform(bowized)

    for i in range(n):
        plt.annotate('+', lowdim[i], fontsize='large', fontweight='bold', color='blue')
    for i in range(n, 2 * n):
        plt.annotate('-', lowdim[i], fontsize='large', fontweight='bold', color='red')
    plt.annotate('?', lowdim[-1], fontsize='large', fontweight='bold', color='cyan')

    plt.xlim(lowdim[:, 0].min() - .1, lowdim[:, 0].max() + .1)
    plt.ylim(lowdim[:, 1].min() - .1, lowdim[:, 1].max() + .1)
    plt.axis('off')


Divergences
-----------

There's a wide class of machine learning algorithms which can run based only
on a particular kind of similarity function between objects,
known as `kernel methods <https://en.wikipedia.org/wiki/Kernel_method>`_.
The best-known of these is the support vector machine (:ref:`SVM <sklearn:svm>`), 
but there are many others as well.
So, if we can come up with a *kernel* between sets,
we can run our SVMs or other kernel methods directly on the input sets.

One way to do that goes like this:
Suppose that the sets we're given are `independent and identically distributed (iid) <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_
samples from some unknown probability distribution.
Then, we can use these samples to statistically estimate a "distance" between those underlying probability distributions.
Some of the options for these distances are:

* the `Kullback–Leibler (KL) divergence <https://en.wikipedia.org/wiki/KL_divergence>`_
* the `Rényi  divergence <https://en.wikipedia.org/wiki/Renyi_divergence#R.C3.A9nyi_divergence>`_
* the :math:`L_2` distance: :math:`\int (p(x) - q(x))^2 dx` where :math:`p` and :math:`q` are the probability densities
* the `Jensen–Shannon divergence <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_
* the `Hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_

We can estimate all of these distances with
:class:`divergences.KNNDivergenceEstimator`.
To turn these distances into a kernel,
we can put it in a :class:`sklearn.pipeline.Pipeline` with
:class:`kernels.PairwisePicker`,
:class:`kernels.Symmetrize`,
:class:`kernels.RBFize`,
:class:`kernels.ProjectPSD`
and then put it in an SVM::

    >>> from sklearn.pipeline import Pipeline
    >>> from skl_groups.divergences import KNNDivergenceEstimator
    >>> from skl_groups.kernels import PairwisePicker, Symmetrize, RBFize, ProjectPSD
    >>> model = Pipeline([
        ('divs', KNNDivergenceEstimator(div_funcs=['kl'], Ks=[2])),
        ('pick', PairwisePicker((0, 0))),
        ('symmetrize', Symmetrize()),
        ('rbf', RBFize(gamma=1, scale_by_median=True)),
        ('project', ProjectPSD()),
        ('svm', SVC(C=1, kernel='precomputed', probability=True)),
    ])
    >>> model.fit(feats[:6], labels[:6]).predict(feats[6:])
    array([False], dtype=bool)

In practice, you definitely want to try tuning the various parameters
with :class:`sklearn.grid_search.GridSearchCV` or similar.
:class:`kernels.RBFize`'s ``gamma`` and :class:`sklearn.svm.SVC`'s ``C`` need it for sure,
probably also the ``Ks`` and ``div_funcs`` in :class:`divergences.KNNDivergenceEstimator`)
(possibly by passing lists there and tuning over different indices
in :class:`kernels.PairwisePicker`).
Making this process more efficient and less work for the user is
`high on the todo list <https://github.com/dougalsutherland/skl-groups/issues/15>`_.

Here's what the divergence matrix looks like for our example data:

.. plot::
    :context:

    plt.close('all')

    from skl_groups.divergences import KNNDivergenceEstimator
    knn_div = KNNDivergenceEstimator(div_funcs=['kl'], Ks=[2])
    divs = knn_div.fit_transform(feats)

    plt.matshow(divs[0, 0], cmap='hot')
    plt.colorbar()

You can see that the distances are pretty low
among the block of the first three positives
and the block of the next four negatives,
and fairly high between the two blocks.
Here's a low-dimensional version of the space that the SVM sees:

.. plot::
    :context:

    plt.close('all')

    from skl_groups import kernels
    K = kernels.Symmetrize().fit_transform(divs[0, 0])
    K = kernels.RBFize(gamma=1, scale_by_median=True).fit_transform(K)
    K = kernels.ProjectPSD().fit_transform(K)
    hidim = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))

    from sklearn.manifold import LocallyLinearEmbedding
    lle = LocallyLinearEmbedding(n_components=2)
    lowdim = lle.fit_transform(hidim)

    for i in range(n):
        plt.annotate('+', lowdim[i], fontsize='large', fontweight='bold', color='blue')
    for i in range(n, 2 * n):
        plt.annotate('-', lowdim[i], fontsize='large', fontweight='bold', color='red')
    plt.annotate('?', lowdim[-1], fontsize='large', fontweight='bold', color='cyan')

    plt.xlim(lowdim[:, 0].min() - .1, lowdim[:, 0].max() + .1)
    plt.ylim(lowdim[:, 1].min() - .1, lowdim[:, 1].max() + .1)
    plt.axis('off')


L2 density transformer
----------------------

:class:`divergences.KNNDivergenceEstimator` can get computationally expensive
if you have a lot of points in your sets or if you have a lot of sets.
If your source data is low-dimensional (no more than 5),
one way around that problem is to use :class:`summaries.L2DensityTransformer`,
which transforms bags of features into vectors whose inner product
is an estimate of the :math:`L_2` distance between the underlying densities.

Thus you can use its output like a regular feature matrix,
which will be roughly equivalent to directly using the output of
:class:`divergences.KNNDivergenceEstimator(div_funcs=['l2'])` as a kernel matrix.
You could also pass it through, say, an RBF kernel.

The way the math here works, your input features all need to be in [0, 1];
:class:`preprocessing.BagMinMaxScaler` can help with that.

For example::

    >>> from skl_groups.preprocessing import BagMinMaxScaler
    >>> from skl_groups.summaries import L2DensityTransformer
    >>> model = Pipeline([
        ('scale', BagMinMaxScaler((0, 1))),
        ('l2', L2DensityTransformer(smoothness=5)),
    ])
    >>> l2ized = model.fit_transform(feats)
    >>> l2ized.shape
    (7, 26)
    >>> SVC().fit(l2ized[:6], labels[:6]).predict(l2ized[6:])
    array([False], dtype=bool)

Note that the intermediate dimensionality is 26.
We can again plot a two-dimensional approximation:

.. plot::
    :context:

    plt.close('all')

    from skl_groups.preprocessing import BagMinMaxScaler
    from skl_groups.summaries import L2DensityTransformer
    
    scaled = BagMinMaxScaler((0, 1)).fit_transform(feats)
    l2ized = L2DensityTransformer(smoothness=5).fit_transform(scaled)

    from sklearn.manifold import LocallyLinearEmbedding
    lle = LocallyLinearEmbedding(n_components=2)
    lowdim = lle.fit_transform(l2ized)

    for i in range(n):
        plt.annotate('+', lowdim[i], fontsize='large', fontweight='bold', color='blue')
    for i in range(n, 2 * n):
        plt.annotate('-', lowdim[i], fontsize='large', fontweight='bold', color='red')
    plt.annotate('?', lowdim[-1], fontsize='large', fontweight='bold', color='cyan')

    plt.xlim(lowdim[:, 0].min() - .1, lowdim[:, 0].max() + .1)
    plt.ylim(lowdim[:, 1].min() - .1, lowdim[:, 1].max() + .1)
    plt.axis('off')
