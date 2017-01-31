#!/bin/bash

if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        make lint || exit -1
        echo "Check documentations of c++ code..."
        make doc 2>log.txt
        (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
        echo "---------Error Log----------"
        cat logclean.txt
        echo "----------------------------"
        (cat logclean.txt|grep warning) && exit -1
        (cat logclean.txt|grep error) && exit -1
    fi
fi

cp make/config.mk config.mk
echo "USE_CUDA=0" >> config.mk

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo "USE_OPENCL=1" >> config.mk
else
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
    echo "USE_OPENCL=0" >> config.mk
fi

if [ ${TASK} == "cpp_test" ] || [ ${TASK} == "all_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    make test || exit -1
    for test in tests/cpp/*_test; do
        ./$test || exit -1
    done
fi

if [ ${TASK} == "python_test" ] || [ ${TASK} == "all_test" ]; then
    make all || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        python -m nose -v tests/python/unittest || exit -1
        python3 -m nose -v tests/python/unittest || exit -1
    else
        nosetests -v tests/python/unittest || exit -1
        nosetests3 -v tests/python/unittest || exit -1
    fi
fi
