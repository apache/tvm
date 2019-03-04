#!/bin/bash

set -e
set -u
set -o pipefail

# install libraries for python package on ubuntu
pip2 install nose pylint==1.9.4 six numpy nose-timer cython decorator scipy tornado typing antlr4-python2-runtime attrs
pip3 install nose pylint==1.9.4 six numpy nose-timer cython decorator scipy tornado typed_ast pytest mypy orderedset antlr4-python3-runtime attrs
