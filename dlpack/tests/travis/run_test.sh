#!/bin/bash
if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        ./tests/scripts/task_lint.sh || exit -1
  fi
fi

if [ ${TASK} == "build" ] || [ ${TASK} == "all_test" ]; then
    ./tests/scripts/task_build.sh || exit -1
fi

echo "All travis test passed.."
