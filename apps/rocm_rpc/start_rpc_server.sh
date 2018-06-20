#/bin/bash
PROJ_ROOT=$(realpath $(dirname "$0")/../..)
export PYTHONPATH=${PROJ_ROOT}/python:${PYTHONPATH}

python -m tvm.exec.rpc_server "$@" --load-library=${PROJ_ROOT}/apps/rocm_rpc/lib/libtvm_runtime_rocm.so
