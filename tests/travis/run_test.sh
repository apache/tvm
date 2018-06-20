#!/bin/bash

if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        ./tests/scripts/task_lint.sh || exit -1
    fi
fi

cp make/config.mk config.mk
echo "USE_CUDA=0" >> config.mk
echo "USE_RPC=1" >> config.mk

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo "USE_OPENCL=1" >> config.mk
    echo "USE_METAL=1" >> config.mk
else
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
    echo "USE_OPENCL=0" >> config.mk
fi

if [ ${TASK} == "verilog_test" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        make -f tests/scripts/packages.mk iverilog
        make all || exit -1
        ./tests/scripts/task_verilog_test.sh || exit -1
    fi
fi

if [ ${TASK} == "cpp_test" ] || [ ${TASK} == "all_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    ./tests/scripts/task_cpp_unittest.sh || exit -1
    ./tests/scripts/task_cpp_topi.sh || exit -1
fi

if [ ${TASK} == "python_test" ] || [ ${TASK} == "all_test" ]; then
    make all || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        ./tests/scripts/task_python_unittest.sh || exit -1
    else
        nosetests -v tests/python/unittest || exit -1
        nosetests3 -v tests/python/unittest || exit -1
    fi
fi
