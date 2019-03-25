#!/bin/bash

set -e
set -u

mkdir -p docs/_build/html
rm -rf docs/_build/html/jsdoc
rm -rf docs/_build/html/javadoc

# remove stale tutorials and always build from scratch.
rm -rf docs/tutorials

# C++ doc
make doc

# JS doc
jsdoc web/tvm_runtime.js web/README.md
mv out docs/_build/html/jsdoc

# Java doc
make javadoc
mv jvm/core/target/site/apidocs docs/_build/html/javadoc

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

cd docs
PYTHONPATH=`pwd`/../python make html
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
