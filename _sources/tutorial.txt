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

.. image :: _static/set-example.png

The test set obviously belongs to the "-" class.
How can we get that answer with machine learning?


Feature representation
----------------------

In scikit-learn, where objects are represented by feature vectors,
most learning methods take in a 2d array
(:class:`numpy.ndarray` or sometimes :mod:`scipy.sparse`)
of shape ``(n, n_features)``
where each of the ``n`` rows represents a different vector.

In skl-groups, each object is instead a set of vectors.
We represent that as a Python :class:`list` of arrays,
where each array is of shape ``(n_pts[i], n_features)``.
The feature dimensionality must be consistent between different sets,
but the number of points in each set can vary.
(This is why we use a list; numpy arrays don't support "ragged" arrays like this.)

Internally, most methods convert these lists into
:class:`features.Features` objects,
which provide a convenient way to access the data consistently,
and can optionally store any metadata you like along with each bag.


Means
-----

Perhaps the most obvious way to summarize a set of vectors is their mean,
which we can get with :class:`summaries.BagMean`.
BagMean will turn a list of features into a single array,
which we can then plug into our favorite :ref:`sklearn classifier <sklearn:supervised-learning>` (or regressor, clusterer, whatever).

Unfortunately, in our toy example, the mean isn't enough to distinguish
between the two classes.

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

.. todo:: visual example with the image

We can perform this in skl_groups with the :class:`summaries.BagOfWords` estimator.


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
To use a kernel, we can put it in a :class:`sklearn.pipeline.Pipeline`
with
:class:`kernels.PairwisePicker`,
:class:`kernels.Symmetrize`,
:class:`kernels.RBFize`,
:class:`kernels.ProjectPSD`
and then put it in an SVM.

.. todo:: Making this less work for the user is high-priority.


L2 density transformer
----------------------

:class:`divergences.KNNDivergenceEstimator` can get computationally expensive
if you have a lot of points in your sets or if you have a lot of sets.
If your source data is low-dimensional (no more than 5),
one way around that problem is to use :class:`summaries.L2DensityTransformer`,
which transforms bags of features into vectors whose inner product
is an estimate of the :math:`L_2` distance between the underlying densities.
