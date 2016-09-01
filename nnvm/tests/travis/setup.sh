#!/bin/bash


if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew install python3
    if [ ${TASK} == "python_test" ]; then
        python -m pip install nose numpy cython --user `whoami`
        python3 -m pip install nose numpy cython --user `whoami`
    fi
fi

if [ ${TASK} == "lint" ]; then
    pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6' --user `whoami`
fi
