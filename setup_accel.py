#!/usr/bin/env python

import os

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


def cython_ext(extension, **kw):
    assert len(extension.sources) == 1
    base, ext = os.path.splitext(extension.sources[0])
    # setuptools sometimes "nicely" turns .pyx into .c for us
    assert ext in {'.pyx', '.c'}
    pyx_path = base + '.pyx'
    c_path = base + '.c'

    try:
        from Cython.Build import cythonize
    except ImportError:
        try:
            pyx_time = os.path.getmtime(pyx_path)
            c_time = os.path.getmtime(c_path)
            if pyx_time > c_time:
                import datetime
                msg = "{pyx_name} file has mtime {pyx_t}, {c_name} has {c_t}"
                raise ValueError(msg.format(
                    pyx_name=os.path.basename(pyx_path),
                    c_name=os.path.basename(c_path),
                    pyx_t=datetime.datetime.fromtimestamp(pyx_time),
                    c_t=datetime.datetime.fromtimestamp(c_time),
                ))
        except (OSError, ValueError) as e:
            msg = "{} extension needs to be compiled, but cython not available:"
            raise ImportError(msg.format(extension.name) + '\n' + str(e))
        else:
            extension.sources[0] = c_path
            return extension
    else:
        return cythonize([extension])[0]


# TODO: don't do this for egg_info, etc
import numpy
from cyflann.extensions import FLANNExtension, build_ext_flann

ext_modules = [
    cython_ext(FLANNExtension('skl_groups_accel.knn_divs',
                              ['skl_groups_accel/knn_divs.pyx'],
                              include_dirs=[numpy.get_include()],
                              extra_compile_args=['-fopenmp'],
                              extra_link_args=['-fopenmp'])),
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
    cmdclass={'build_ext': build_ext_flann},
    install_requires=[
        'cyflann >= 0.1.22',
    ],
    zip_safe=False,
)
