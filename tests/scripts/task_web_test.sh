#!/bin/bash
export PYTHONPATH=python

cp /emsdk-portable/.emscripten ~/.emscripten
source /emsdk-portable/emsdk_env.sh

export EM_CONFIG=${HOME}/.emscripten
export EM_CACHE=${HOME}/.emscripten_cache

echo "Build TVM Web runtime..."
make web

echo "Prepare test libraries..."
python tests/web/prepare_test_libs.py || exit -1

echo "Start testing..."

for test in tests/web/test_*.js; do
    echo node $test
    node $test || exit -1
done

echo "All tests finishes..."
