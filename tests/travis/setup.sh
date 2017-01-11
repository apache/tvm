#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    if [ ${TASK} == "python_test" ] || [ ${TASK} == "all_test" ]; then
        brew update
        brew install python3
        python -m pip install --user nose numpy
        python3 -m pip install --user nose numpy
    fi
fi

if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        pip install --user cpplint 'pylint==1.4.4' 'astroid==1.3.6'
    fi
fi
