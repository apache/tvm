#!/bin/bash

sgx_sdk=${SGX_SDK:=/opt/sgxsdk}

export LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH}
export CC=clang-6.0
export AR=llvm-ar-6.0
export TVM_CACHE_DIR=/tmp

make && printf "\n" && python3 run_model.py
