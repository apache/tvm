#!/bin/bash
cd docs
PYTHONPATH=../python make html || exit -1
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
