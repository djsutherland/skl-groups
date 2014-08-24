#!/bin/bash
# the Travis "script" step: see http://docs.travis-ci.com/ and ../.travis.yml
set -e

cd $HOME # get out of source directory to avoid confusing nose

source activate without-flann
nosetests --exe skl_groups
source deactivate

for env in with-pyflann with-cyflann with-accel; do
    source activate $env
    nosetests --exe skl_groups.tests.test_divs_knn
    source deactivate
done
