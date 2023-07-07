
set -e
set -u
set -o pipefail
set -x

TVM_DIR=/home/sergei/projects/MIR/TVM/tvm
TVM_BUILD_DIR=/home/sergei/projects/MIR/TVM/build-Release-19.06.23

ENV_VARS="--env PYTHONPATH=${TVM_DIR}/python"
ENV_VARS="${ENV_VARS} --env TVM_HOME=${TVM_DIR}"
ENV_VARS="${ENV_VARS} --env TVM_LIBRARY_PATH=${TVM_BUILD_DIR}"
ENV_VARS="${ENV_VARS} --env TVM_CONFIGS_JSON_DIR=${TVM_DIR}/configs/host"
ENV_VARS="${ENV_VARS} --env CRT_ROOT=${TVM_BUILD_DIR}/standalone_crt"

pushd ${TVM_DIR}
./docker/bash.sh --mount ${TVM_BUILD_DIR} ${ENV_VARS} ci_cortexm
