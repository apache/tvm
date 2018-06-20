#!/bin/bash

sgx_sdk=${SGX_SDK:=/opt/sgxsdk}

make
echo "========================="
LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH} bin/run_model
