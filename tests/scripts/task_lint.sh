#!/bin/bash

cleanup()
{
  rm -rf /tmp/$$.*
}
trap cleanup 0

echo "Check codestyle of c++ code..."
make cpplint || exit -1
echo "Check codestyle of python code..."
make pylint || exit -1
echo "Check codestyle of jni code..."
make jnilint || exit -1
echo "Check documentations of c++ code..."
make doc 2>/tmp/$$.log.txt

grep -v -E "ENABLE_PREPROCESSING|unsupported tag" < /tmp/$$.log.txt > /tmp/$$.logclean.txt
echo "---------Error Log----------"
cat /tmp/$$.logclean.txt
echo "----------------------------"
grep -E "warning|error" < /tmp/$$.logclean.txt && exit -1
exit 0
