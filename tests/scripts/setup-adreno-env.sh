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


ENVIRONMENT=""
RPC_PORT=""
ADB_SERIAL=""
LISTEN_PORT=5000

function usage() {
    echo "Helper script to setup the environment for Tracker, RPC Device and for application"
    echo "Usage (Help) : source setup-adreno-env.sh -h"
    echo "Usage (Tracker): source setup-adreno-env.sh -e tracker -p <RPC PORT>"
    echo "Usage (Device): source setup-adreno-env.sh -e device -p <RPC PORT> -d <Android Serial> [-l <device listen port>]"
    echo "Usage (Query): source setup-adreno-env.sh -e query -p <RPC PORT>"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--environment)
      ENVIRONMENT="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--rpc-port)
      RPC_PORT="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--android-device)
      ADB_SERIAL="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--listen-port)
      LISTEN_PORT="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      usage
      return 0
      ;;
    -*|--*)
      usage
      return 0
      ;;
    *)
      ;;
  esac
done

echo "ENVIRONMENT   = ${ENVIRONMENT}"
echo "RPC_PORT      = ${RPC_PORT}"
echo "ADB_SERIAL    = ${ADB_SERIAL}"
echo "DEVICE LISTEN POPRT    = ${LISTEN_PORT}"


function def_environment() {
    source tests/scripts/setup-pytest-env.sh
    export LD_LIBRARY_PATH="${TVM_PATH}/build:${LD_LIBRARY_PATH}"
    export TVM_TRACKER_HOST=0.0.0.0
    export TVM_TRACKER_PORT=$RPC_PORT
    export RPC_DEVICE_KEY="android"
    export RPC_TARGET="adreno"
    export TVM_NDK_CC="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
    # Compiler definition for c-runtime while empty mod (llvm -mtriple ineffective here).
    export CXX="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
}

def_environment

case ${ENVIRONMENT} in

  "tracker")
    echo "Starting Tracker on port :${TVM_TRACKER_PORT}"
    def_environment
    python3 -m tvm.exec.rpc_tracker --host "${TVM_TRACKER_HOST}" --port "${TVM_TRACKER_PORT}"
    ;;

  "device")
    echo "Running RPC on device : ${ADB_SERIAL} with key $RPC_DEVICE_KEY"
    def_environment
    export ANDROID_SERIAL=${ADB_SERIAL}

    TARGET_FOLDER=/data/local/tmp/tvm_ci-${USER}
    CPP_LIB=`find ${ANDROID_NDK_HOME} -name libc++_shared.so | grep aarch64`
    adb shell "mkdir -p ${TARGET_FOLDER}"
    adb push build-adreno-target/tvm_rpc ${TARGET_FOLDER}/tvm_rpc-${USER}
    adb push build-adreno-target/libtvm_runtime.so ${TARGET_FOLDER}
    if [ -f ${CPP_LIB} ] ; then
        adb push ${CPP_LIB} ${TARGET_FOLDER}
    fi

    adb reverse tcp:${TVM_TRACKER_PORT} tcp:${TVM_TRACKER_PORT}
    adb forward tcp:${LISTEN_PORT} tcp:${LISTEN_PORT}
    adb forward tcp:$((LISTEN_PORT + 1)) tcp:$((LISTEN_PORT + 1))
    adb forward tcp:$((LISTEN_PORT + 2)) tcp:$((LISTEN_PORT + 2))
    adb forward tcp:$((LISTEN_PORT + 3)) tcp:$((LISTEN_PORT + 3))
    adb shell "cd ${TARGET_FOLDER}; killall -9 tvm_rpc-${USER}; sleep 2; export CLML_PROFILING=1; export CLML_IS_TUNING_RUN=1; export CLML_TUNING_CACHE=clml.bin; LD_LIBRARY_PATH=${TARGET_FOLDER}/ ./tvm_rpc-${USER} server --host=0.0.0.0 --port=${LISTEN_PORT} --port-end=$((LISTEN_PORT + 10)) --tracker=127.0.0.1:${TVM_TRACKER_PORT} --key=${RPC_DEVICE_KEY}"
    ;;

  "query")
    def_environment
    echo "Setting dev environment with Tracker Port : $TVM_TRACKER_HOST} and the available devices are"
    python3 -m tvm.exec.query_rpc_tracker --port ${TVM_TRACKER_PORT}
    ;;

  *)
    usage
    ;;
esac
