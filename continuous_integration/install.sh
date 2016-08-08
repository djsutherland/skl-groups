#!/bin/bash
# the Travis "install" step: see http://docs.travis-ci.com/ and ../.travis.yml
set -e

os=$(uname)
if [[ "$os" == "Linux" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
        -O miniconda.sh
elif [[ "$os" == "Darwin" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh \
        -O miniconda.sh
else
    echo "unknown os '$os'"
    exit 1
fi
chmod +x miniconda.sh
./miniconda.sh -b
export PATH="$HOME/miniconda/bin:$PATH"

conda update --yes --quiet conda

PKGS="python=$PYTHON_VERSION pip nose setuptools testfixtures cython "
PKGS="$PKGS numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"
PKGS="$PKGS scikit-learn=$SKLEARN_VERSION versiontools anaconda-client"

conda create --yes -n without-flann -c dougal \
    $PKGS

# temporarily don't do with-pyflann (#24)
# conda create --yes -n with-pyflann -c https://conda.binstar.org/dougal \
#     $PKGS pyflann=$PYFLANN_VERSION

conda create --yes -n with-cyflann -c dougal \
    $PKGS cyflann=$CYFLANN_VERSION

conda create --yes -n with-accel -c dougal \
    $PKGS cyflann=$CYFLANN_VERSION


# temporarily don't do with-pyflann (#24)
for env in without-flann with-cyflann with-accel; do
    source activate $env
    python setup.py install
    source deactivate
done

source activate with-accel
python setup_accel.py install
source deactivate
