#!/bin/bash
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

make cpptest -j8 || exit -1
for test in build/*_test; do
    ./$test || exit -1
done
