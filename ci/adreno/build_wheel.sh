#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}


AUDITWHEEL_OPTS="--exclude libLLVM --plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
if [[ ${GPU} == cuda* ]]; then
    AUDITWHEEL_OPTS="--exclude libcuda --exclude libcudart --exclude libnvrtc --exclude libcublas --exclude libcublasLt  ${AUDITWHEEL_OPTS}"
fi

cd ${WORKSPACE_CWD}/python && python setup.py bdist_wheel

rm -rf ${WORKSPACE_CWD}/wheels/
auditwheel repair ${AUDITWHEEL_OPTS} dist/*.whl
mv ${WORKSPACE_CWD}/python/repaired_wheels/ ${WORKSPACE_CWD}/wheels/

chown -R $ENV_USER_ID:$ENV_GROUP_ID ${WORKSPACE_CWD}/wheels/
