#!/bin/bash


if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    echo "Check documentations of c++ code..."
    make doc 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
    echo "---------Error Log----------"
    cat logclean.txt
    echo "----------------------------"
    (cat logclean.txt|grep warning) && exit -1
    (cat logclean.txt|grep error) && exit -1
    exit 0
fi


if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
fi

if [ ${TASK} == "cpp_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make test || exit -1
    for test in tests/cpp/*_test; do
        ./$test || exit -1
    done
    exit 0
fi

# run two test one for cython, one for ctypes
if [ ${TASK} == "python_test" ]; then
    make clean
    make -j all || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        python -m nose tests/python/unittest/ || exit -1
        python3 -m nose tests/python/unittest/ || exit -1
    else
        nosetests tests/python/unittest/ || exit -1
        nosetests3 tests/python/unittest/ || exit -1
    fi

    make cython || exit -1
    make cython3 || exit -1

    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        python -m nose tests/python/unittest/ || exit -1
        python3 -m nose tests/python/unittest/ || exit -1
    else
        nosetests tests/python/unittest/ || exit -1
        nosetests3 tests/python/unittest/ || exit -1
    fi
    exit 0
fi
