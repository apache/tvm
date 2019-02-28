#!/bin/bash
echo "Check codestyle of c++ code..."
make cpplint || exit -1
echo "Check codestyle of python code..."
make pylint || exit -1
echo "Check codestyle of jni code..."
make jnilint || exit -1
echo "Check documentations of c++ code..."
make doc 2>log.txt
grep -v -E "ENABLE_PREPROCESSING|unsupported tag" < log.txt > logclean.txt
echo "---------Error Log----------"
cat logclean.txt
echo "----------------------------"
grep -E "warning|error" < logclean.txt && exit -1
rm logclean.txt
rm log.txt
