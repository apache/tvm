#!/usr/bin/env bash
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
