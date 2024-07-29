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

set -euxo pipefail

export TVM_TEST_TARGETS="opencl"
export TVM_RELAY_OPENCL_TEXTURE_TARGETS="opencl -device=adreno"

source tests/scripts/setup-pytest-env.sh
export PYTHONPATH=${PYTHONPATH}:${TVM_PATH}/apps/extension/python
export LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH:-}"
export TVM_INTEGRATION_TESTSUITE_NAME=python-integration-adreno

sudo apt install net-tools -y

find_free_port () {
    port=$1
    RANGE=$2
    isfree=0
    while [ 0 -eq "$isfree" ]; do
        for ii in `seq 0 $RANGE` ; do
            sport=$((port+ii))
            netstat -taln | grep LISTEN | grep $sport > /dev/null
            isfree=$?
            if [ 0 -eq "$isfree" ] ; then
                port=$((port+1))
                break
            fi
        done
        sleep 1
    done
    echo $port
    return 0
}

export TVM_TRACKER_HOST=127.0.0.1
FREE_PORT=`find_free_port 9000 1`
export TVM_TRACKER_PORT=$FREE_PORT
export RPC_DEVICE_KEY="android"
export RPC_TARGET="adreno"
export TVM_NDK_CC="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

env PYTHONPATH=python python3 -m tvm.exec.rpc_tracker --host "${TVM_TRACKER_HOST}" --port "${TVM_TRACKER_PORT}" &
TRACKER_PID=$!
sleep 5   # Wait for tracker to bind

export ANDROID_SERIAL=$1

TARGET_FOLDER=/data/local/tmp/tvm_ci-${USER}-${TVM_TRACKER_PORT}
adb shell "mkdir -p ${TARGET_FOLDER}"
adb push build-adreno-target/tvm_rpc ${TARGET_FOLDER}/tvm_rpc-${USER}-${TVM_TRACKER_PORT}
adb push build-adreno-target/libtvm_runtime.so ${TARGET_FOLDER}
CPP_LIB=`find ${ANDROID_NDK_HOME} -name libc++_shared.so | grep aarch64`
if [ -f ${CPP_LIB} ] ; then
    adb push ${CPP_LIB} ${TARGET_FOLDER}
fi
if [ -f "${ADRENOACCL_SDK}/include/adrenoaccl.h" ] ; then
    adb push ${ADRENOACCL_SDK}/lib/libadrenoaccl.so ${TARGET_FOLDER}
fi

adb reverse tcp:${TVM_TRACKER_PORT} tcp:${TVM_TRACKER_PORT}
ADB_PORTS_RANGE=4
RPC_LISTEN_PORT=`find_free_port 6000 ${ADB_PORTS_RANGE}`
export DEVICE_LISTEN_PORT=${RPC_LISTEN_PORT}
for ii in `seq 0 ${ADB_PORTS_RANGE}` ; do
  adb forward tcp:$((RPC_LISTEN_PORT+ii)) tcp:$((RPC_LISTEN_PORT+ii))
done
env adb shell "cd ${TARGET_FOLDER}; killall -9 tvm_rpc-${USER}-${TVM_TRACKER_PORT}; sleep 2; TARGET_RPC_TMP=${TARGET_FOLDER}/rpc_tmp LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./tvm_rpc-${USER}-${TVM_TRACKER_PORT} server --host=0.0.0.0 --port=${RPC_LISTEN_PORT} --port-end=$((RPC_LISTEN_PORT+${ADB_PORTS_RANGE})) --tracker=127.0.0.1:${TVM_TRACKER_PORT} --key=${RPC_DEVICE_KEY}" &
DEVICE_PID=$!
sleep 5 # Wait for the device connections
clean_ports() {
    for ii in `seq 0 ${ADB_PORTS_RANGE}` ; do
        adb forward --remove tcp:$((RPC_LISTEN_PORT+${ii})) || true
    done;
}
trap "{ kill ${TRACKER_PID} || true; kill ${DEVICE_PID} || true; clean_ports; cleanup;}" 0

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f
# Test TVM
make cython3

# The RPC to remote Android device has issue of hang after few tests with in CI environments.
# Lets run them individually on fresh rpc session.
# OpenCL texture test on Adreno
TEXTURE_TESTS=$(./ci/scripts/jenkins/pytest_ids.py --folder tests/python/relay/opencl_texture)
i=0
IFS=$'\n'
for node_id in $TEXTURE_TESTS; do
    echo "$node_id"
    run_pytest ctypes "$TVM_INTEGRATION_TESTSUITE_NAME-opencl-texture-$i" "$node_id" --reruns=0
    i=$((i+1))
done

# Adreno CLML test
CLML_TESTS=$(./ci/scripts/jenkins/pytest_ids.py --folder tests/python/contrib/test_clml)
i=0
for node_id in $CLML_TESTS; do
    echo "$node_id"
    CXX=${TVM_NDK_CC} run_pytest ctypes "$TVM_INTEGRATION_TESTSUITE_NAME-openclml-$i" "$node_id" --reruns=0
    i=$((i+1))
done

kill ${TRACKER_PID} || true
kill ${DEVICE_PID} || true
clean_ports
