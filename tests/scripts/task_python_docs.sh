#!/bin/bash
mkdir -p docs/_build/html
rm -rf docs/_build/html/jsdoc

# remove stale tutorials and always build from scratch.
rm -rf docs/tutorials

# C++ doc
make doc

# JS doc
jsdoc web/tvm_runtime.js web/README.md || exit -1
mv out docs/_build/html/jsdoc || exit -1

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

cd docs
PYTHONPATH=`pwd`/../python make html || exit -1
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
