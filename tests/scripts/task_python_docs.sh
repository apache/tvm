#!/bin/bash
cd docs
PYTHONPATH=../python make html
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
