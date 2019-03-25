#!/bin/bash

set -e
set -u

export PYTHONPATH=python
make verilog
nosetests -v tests/verilog/unittest
nosetests -v tests/verilog/integration
