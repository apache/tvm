#!/bin/bash

set -e
set -u
set -o pipefail

cleanup()
{
  rm -rf /tmp/$$.*
}
trap cleanup 0

echo "Check codestyle of c++ code..."
make cpplint
echo "Check codestyle of python code..."
make pylint
echo "Check codestyle of jni code..."
make jnilint
echo "Check documentations of c++ code..."
make doc 2>/tmp/$$.log.txt

grep -v -E "ENABLE_PREPROCESSING|unsupported tag" < /tmp/$$.log.txt > /tmp/$$.logclean.txt || true
echo "---------Error Log----------"
cat /tmp/$$.logclean.txt
echo "----------------------------"
grep -E "warning|error" < /tmp/$$.logclean.txt || true
