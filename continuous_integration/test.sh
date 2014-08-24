#!/bin/bash
# the Travis "script" step: see http://docs.travis-ci.com/ and ../.travis.yml
set -e

source activate without-flann
nosetests --exe skl-groups
source deactivate

for env in with-pyflann with-cyflann with-accel; do
    source activate $env
    nosetests --exe skl-groups.tests.test_divs_knn
    source deactivate
done
