#!/bin/bash

set -e
set -u

export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH:-}"

make cpptest -j8
for test in build/*_test; do
    ./$test
done
