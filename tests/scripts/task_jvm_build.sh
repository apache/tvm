#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Build the TVM JVM package (tvm4j) using Maven with platform detection.
# Usage: task_jvm_build.sh [goal] [mvn-extra-args...]
#
# The first positional argument sets the Maven goal (default: "clean package").
# Extra arguments after the goal are passed through to mvn, e.g.:
#   task_jvm_build.sh install
#   task_jvm_build.sh "clean package" -DskipTests=false -Dtest.tempdir=/tmp/foo

set -euxo pipefail

GOAL="${1:-clean package}"
if [ "$#" -gt 0 ]; then
  shift
fi

ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
TVM_BUILD_PATH="${TVM_BUILD_PATH:-${ROOTDIR}/build}"
TVM_BUILD_PATH="$(realpath "${TVM_BUILD_PATH}")"

DLPACK_PATH="${DLPACK_PATH:-${ROOTDIR}/3rdparty/tvm-ffi/3rdparty/dlpack}"

INCLUDE_FLAGS="-I${ROOTDIR}/include -I${DLPACK_PATH}/include"
PKG_CFLAGS="-Wall -O3 ${INCLUDE_FLAGS} -fPIC"
PKG_LDFLAGS=""

if [ "$(uname -s)" = "Darwin" ]; then
    JVM_PKG_PROFILE="osx-x86_64"
    SHARED_LIBRARY_SUFFIX="dylib"
elif [ "${OS:-}" = "Windows_NT" ]; then
    JVM_PKG_PROFILE="windows"
    SHARED_LIBRARY_SUFFIX="dll"
else
    JVM_PKG_PROFILE="linux-x86_64"
    SHARED_LIBRARY_SUFFIX="so"
fi

JVM_TEST_ARGS="${JVM_TEST_ARGS:--DskipTests -Dcheckstyle.skip=true}"

cd "${ROOTDIR}/jvm"
# shellcheck disable=SC2086
mvn ${GOAL} \
    "-P${JVM_PKG_PROFILE}" \
    "-Dcxx=${CXX:-g++}" \
    "-Dcflags=${PKG_CFLAGS}" \
    "-Dldflags=${PKG_LDFLAGS}" \
    "-Dcurrent_libdir=${TVM_BUILD_PATH}" \
    ${JVM_TEST_ARGS} \
    "$@"
