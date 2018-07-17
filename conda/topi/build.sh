#!/bin/bash

set -e

cd topi/python
$PYTHON setup.py install --single-version-externally-managed --record=/tmp/record.txt
