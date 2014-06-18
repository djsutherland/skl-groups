#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = "Compiled components to speed up skl-groups."
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
NAME = "skl-groups-accel"
AUTHOR = MAINTAINER = "Dougal J. Sutherland"
AUTHOR_EMAIL = MAINTAINER_EMAIL = "dsutherl@cs.cmu.edu"
URL = DOWNLOAD_URL = "https://github.com/dougalsutherland/skl-groups-accel"
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

import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# Hack: set a global variable to tell the main __init__ not to load components
# that aren't built yet.
builtins.__SKL_GROUPS_ACCEL_SETUP__ = True


import skl_groups_accel
VERSION = skl_groups_accel.__version__


setup(
    name=NAME,
    version=VERSION,
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
        'skl_groups_accel',
    ],
    install_requires=[
        'skl-groups',
        'cyflann >= 0.1.15',
    ],
    zip_safe=False,
)
