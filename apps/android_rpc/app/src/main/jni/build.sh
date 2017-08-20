#!/bin/bash
PATH="$PATH:/usr/local/bin"
CURR_DIR=$(cd `dirname $0`; pwd)
ROOT_DIR="$CURR_DIR/../../../../../.."
javah -o $CURR_DIR/ml_dmlc_tvm_native_c_api.h -cp "$ROOT_DIR/jvm/core/target/*" ml.dmlc.tvm.LibInfo || exit -1
cp -f $ROOT_DIR/jvm/native/src/main/native/ml_dmlc_tvm_native_c_api.cc $CURR_DIR/ || exit -1
cp -f $ROOT_DIR/jvm/native/src/main/native/jni_helper_func.h $CURR_DIR/ || exit -1
rm -rf $CURR_DIR/../libs
ndk-build --directory=$CURR_DIR
