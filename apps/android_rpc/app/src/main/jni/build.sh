#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR="$CURR_DIR/../../../../../.."
javah -o ml_dmlc_tvm_native_c_api.h -cp "$ROOT_DIR/jvm/core/target/*" ml.dmlc.tvm.LibInfo || exit -1
cp $ROOT_DIR/jvm/native/src/main/native/ml_dmlc_tvm_native_c_api.cc . || exit -1
cp $ROOT_DIR/jvm/native/src/main/native/jni_helper_func.h . || exit -1
ndk-build
