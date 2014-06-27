#!/usr/bin/env python

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension

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

setup_args = {}

def cython_ext(name, **kw):
    pyx_path = 'skl_groups_accel/{}.pyx'.format(name)
    c_path = 'skl_groups_accel/{}.c'.format(name)
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        try:
            pyx_time = os.path.getmtime(pyx_path)
            c_time = os.path.getmtime(c_path)
            if pyx_time >= c_time:
                import datetime
                msg = "{name}.pyx file has mtime {pyx_t}, {name}.c has {c_t}"
                raise ValueError(msg.format(
                    name=name,
                    pyx_t=datetime.datetime.fromtimestamp(pyx_time),
                    c_t=datetime.datetime.fromtimestamp(c_time),
                ))
        except (OSError, ValueError) as e:
            msg = "{} extension needs to be compiled, but cython not available:"
            raise ImportError(msg.format(name) + '\n' + str(e))
        else:
            source_file = c_path
    else:
        source_file = pyx_path
        setup_args['cmdclass'] = {'build_ext': build_ext}

    return Extension("skl_groups_accel.{}".format(name), [source_file], **kw)

# TODO: don't do this for egg_info, etc
import numpy
import cyflann
ext_modules = [
    cython_ext('knn_divs',
               include_dirs=[numpy.get_include(), cyflann.get_flann_include()],
               extra_compile_args=['-fopenmp'],
               extra_link_args=['-fopenmp', cyflann.get_flann_lib()])
]

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
    ext_modules=ext_modules,
    install_requires=[
        'skl-groups',
        'cyflann >= 0.1.15',
    ],
    zip_safe=False,
    **setup_args
)
