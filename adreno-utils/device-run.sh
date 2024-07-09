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

DEVICE=$1
TRACKER_PORT=$2
DEVICE_PORT=$3
DEVICE_KEY=$4

adb -s $DEVICE shell "mkdir -p /data/local/tmp/tvm-rpc-$USER"
adb -s $DEVICE push tvm_rpc /data/local/tmp/tvm-rpc-$USER/tvm_rpc-$USER
adb -s $DEVICE push libtvm_runtime.so /data/local/tmp/tvm-rpc-$USER/
CPP_LIB=`find ${ANDROID_NDK} -name libc++_shared.so | grep aarch64`
if [ -f ${CPP_LIB} ] ; then
    adb -s $DEVICE push ${CPP_LIB} /data/local/tmp/tvm-rpc-$USER/
fi
if [ -f ${ADRENOACCL_SDK}/lib/libadrenoaccl.so ] ; then
    adb -s $DEVICE push ${ADRENOACCL_SDK}/lib/libadrenoaccl.so /data/local/tmp/tvm-rpc-$USER/
fi
adb -s $DEVICE reverse tcp:${TRACKER_PORT} tcp:${TRACKER_PORT}
adb -s $DEVICE shell "cd /data/local/tmp/tvm-rpc-$USER/ ; pwd; killall -9 tvm_rpc-$USER; sleep 2; LD_LIBRARY_PATH=/data/local/tmp/tvm-rpc-$USER/ ./tvm_rpc-$USER server --host=0.0.0.0 --port=${DEVICE_PORT} --port-end=$((DEVICE_PORT + 10)) --tracker=127.0.0.1:${TRACKER_PORT} --key=${DEVICE_KEY}"
