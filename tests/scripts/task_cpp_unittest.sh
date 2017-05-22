#!/bin/bash
make test -j8 || exit -1
for test in tests/cpp/*_test; do
    ./$test || exit -1
done
