#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versiontools_support

DESCRIPTION = "Addon to scikit-learn for handling set-based data."
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
NAME = "skl-groups"
AUTHOR = MAINTAINER = "Dougal J. Sutherland"
AUTHOR_EMAIL = MAINTAINER_EMAIL = "dsutherl@cs.cmu.edu"
URL = DOWNLOAD_URL = "https://github.com/dougalsutherland/skl-groups"
LICENSE = "BSD"
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Scientific/Engineering :: Information Analysis",
]


setup(
    name=NAME,
    version=':versiontools:skl_groups',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=[
        'skl_groups',
        'skl_groups.divergences',
        'skl_groups.kernels',
        'skl_groups.summaries',
        'skl_groups.tests',
    ],
    install_requires=['scikit-learn >= 0.17'],
    extras_require={
        'knn_divs': ['cyflann >= 0.1.15'],
    },
    zip_safe=False,
)
