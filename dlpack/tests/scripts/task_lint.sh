#!/bin/bash
mkdir -p bin

if [ ! -f bin/lint.py ]; then
    echo "Grab linter ..."
    wget https://raw.githubusercontent.com/dmlc/dmlc-core/master/scripts/lint.py
    mv lint.py bin/lint.py
fi

echo "Check codestyle of c++ code..."
python bin/lint.py dlpack cpp include contrib

echo "Check doxygen generation..."
make doc 2>log.txt
(cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
echo "---------Error Log----------"
cat logclean.txt
echo "----------------------------"
(cat logclean.txt|grep warning) && exit -1
(cat logclean.txt|grep error) && exit -1
rm logclean.txt
rm log.txt
echo "All checks passed..."
