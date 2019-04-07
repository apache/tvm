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
PATH="$PATH:/usr/local/bin"
CURR_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR="$CURR_DIR/../../../../../.."
javah -o $CURR_DIR/ml_dmlc_tvm_native_c_api.h -cp "$ROOT_DIR/jvm/core/target/*" ml.dmlc.tvm.LibInfo || exit -1
cp -f $ROOT_DIR/jvm/native/src/main/native/ml_dmlc_tvm_native_c_api.cc $CURR_DIR/ || exit -1
cp -f $ROOT_DIR/jvm/native/src/main/native/jni_helper_func.h $CURR_DIR/ || exit -1
rm -rf $CURR_DIR/../libs
ndk-build --directory=$CURR_DIR
