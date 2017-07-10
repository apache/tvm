#!/bin/bash
make jvmpkg JVM_TEST_ARGS="-DskipTests=false" || exit -1
