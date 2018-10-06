#!/bin/bash

sgx_sdk=${SGX_SDK:=/opt/sgxsdk}
LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH} make
printf "\n"
LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH} TVM_CACHE_DIR=/tmp python3 run_model.py
