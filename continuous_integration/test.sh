#!/bin/bash
# the Travis "script" step: see http://docs.travis-ci.com/ and ../.travis.yml
set -e

cd $HOME # get out of source directory to avoid confusing nose

source activate without-flann
nosetests --exe skl_groups
source deactivate

# temporarily don't do with-pyflann (#24)
for env in with-cyflann with-accel; do
    source activate $env
    nosetests --exe skl_groups.tests.test_divs_knn
    source deactivate
done
