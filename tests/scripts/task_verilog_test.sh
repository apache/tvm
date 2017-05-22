#!/bin/bash
export PYTHONPATH=python
make verilog || exit -1
nosetests -v tests/verilog/unittest || exit -1
nosetests -v tests/verilog/integration || exit -1
