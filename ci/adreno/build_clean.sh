#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

rm -rf ${WORKSPACE_CWD}/build/ \
	${WORKSPACE_CWD}/python/dist/ \
	${WORKSPACE_CWD}/python/build/ \
	${WORKSPACE_CWD}/python/__pycache__/ \
	${WORKSPACE_CWD}/python/tvm.egg-info
