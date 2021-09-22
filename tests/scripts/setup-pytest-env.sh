#!/bin/bash
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

# NOTE: allow unbound variable here
set +u

if [[ ! -z $CI_PYTEST_ADD_OPTIONS ]]; then
    export PYTEST_ADDOPTS="-s -v $CI_PYTEST_ADD_OPTIONS $PYTEST_ADDOPTS"
else
    export PYTEST_ADDOPTS="-s -v $PYTEST_ADDOPTS"
fi
set -u

export TVM_PATH=`pwd`
export PYTHONPATH="${TVM_PATH}/python"

export TVM_PYTEST_RESULT_DIR="${TVM_PATH}/build/pytest-results"
mkdir -p "${TVM_PYTEST_RESULT_DIR}"

if [ -n "${CI_CPUSET_NUM_CPUS-}" ]; then
    # When the # of CPUs has been restricted (e.g. when --cpuset-cpus has been passed to docker by
    # docker/bash.sh), explicitly use all available CPUs. This environment variable is set by
    # docker/bash.sh when it sets --cpuset-cpus.
    PYTEST_NUM_CPUS="${CI_CPUSET_NUM_CPUS}"
else
    # Else attempt to use $(nproc) - 1.
    PYTEST_NUM_CPUS=$(nproc)
    if [ -z "${PYTEST_NUM_CPUS}" ]; then
        echo "WARNING: nproc failed; running pytest with only 1 CPU"
        PYTEST_NUM_CPUS=1
    elif [ ${PYTEST_NUM_CPUS} -gt 1 ]; then
        PYTEST_NUM_CPUS=$(expr ${PYTEST_NUM_CPUS} - 1)  # Don't nuke interactive work.
    fi

    # Don't use >4 CPUs--in general, we only use 4 CPUs in testing, so we want to retain this
    # maximum for the purposes of reproducing the CI. You can still override this by setting
    # --cpuset-cpus in docker/bash.sh.
    if [ ${PYTEST_NUM_CPUS} -gt 4 ]; then
        PYTEST_NUM_CPUS=4
    fi
fi

function run_pytest() {
    local extra_args=( )
    if [ "$1" == "--parallel" ]; then
        extra_args=( -n "${PYTEST_NUM_CPUS}" )
        shift
    fi
    local ffi_type="$1"
    shift
    local test_suite_name="$1"
    shift
    if [ -z "${ffi_type}" -o -z "${test_suite_name}" ]; then
        echo "error: run_pytest called incorrectly: run_pytest ${ffi_type} ${test_suite_name} $@"
        echo "usage: run_pytest <FFI_TYPE> <TEST_SUITE_NAME> [pytest args...]"
        exit 2
    fi
    TVM_FFI=${ffi_type} python3 -m pytest \
           --timeout=480 \
           -o "junit_suite_name=${test_suite_name}-${ffi_type}" \
           "--junit-xml=${TVM_PYTEST_RESULT_DIR}/${test_suite_name}-${ffi_type}.xml" \
           "--junit-prefix=${ffi_type}" \
           "${extra_args[@]}" \
           "$@"
}
