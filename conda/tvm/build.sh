#!/bin/bash

set -e

cd python
$PYTHON setup.py install --single-version-externally-managed --record=/tmp/record.txt
