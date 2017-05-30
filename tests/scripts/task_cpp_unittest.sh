#!/bin/bash
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

make test -j8 || exit -1
for test in tests/cpp/*_test; do
    ./$test || exit -1
done
