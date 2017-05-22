#!/bin/bash
echo "Check codestyle of c++ code..."
make cpplint || exit -1
echo "Check codestyle of python code..."
make pylint || exit -1
echo "Check documentations of c++ code..."
make doc 2>log.txt
(cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
echo "---------Error Log----------"
cat logclean.txt
echo "----------------------------"
(cat logclean.txt|grep warning) && exit -1
(cat logclean.txt|grep error) && exit -1
rm logclean.txt
rm log.txt
