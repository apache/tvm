#!/bin/bash
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

set -e
set -u
set -o pipefail

set -x

if [ -z "${TVM_VENV+x}" ]; then
    echo "ERROR: expect TVM_VENV env var to be set"
    exit 2
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: docker/install/ubuntu_install_python.sh <PYTHON_VERSION>"
    exit -1
fi
PYTHON_VERSION=$1

# Install base dependencies required by this script.
apt-get update
apt-install-and-clear -y acl

# Install managed Python to a shared location so both root and the CI runtime user can use it.
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/opt/uv/python}
export UV_PYTHON_INSTALL_DIR
export UV_MANAGED_PYTHON=${UV_MANAGED_PYTHON:-1}
mkdir -p "${UV_PYTHON_INSTALL_DIR}"
chmod 755 "${UV_PYTHON_INSTALL_DIR}"
uv python install "${PYTHON_VERSION}"

# Allow disabling user site-packages, even with sudo; this makes it harder to repro CI failures
# locally because it's hard to tell what might be in this directory.
echo "Defaults env_keep += \"PYTHONNOUSERSITE\"" >/etc/sudoers.d/91-preserve-python-nousersite
export PYTHONNOUSERSITE=1

venv_dir="$(dirname "${TVM_VENV}")"
mkdir -p "${venv_dir}"
uv venv --python "${PYTHON_VERSION}" "${TVM_VENV}"

# NOTE: Only in python3.9 does venv guarantee it creates the python3.X binary.
# This is needed so that CMake's find_package(PythonInterp) works inside the venv.
# See https://bugs.python.org/issue39656
if [ ! -e "${TVM_VENV}/bin/python${PYTHON_VERSION}" ]; then
    ln -s "${TVM_VENV}/bin/python" "${TVM_VENV}/bin/python${PYTHON_VERSION}"
fi

addgroup tvm-venv || true
chgrp -R tvm-venv "${TVM_VENV}"
setfacl -R -d -m group:tvm-venv:rwx "${TVM_VENV}"
setfacl -R -m group:tvm-venv:rwx "${TVM_VENV}"

# Prevent further use of pip3 via the system.
# There may be multiple (i.e. from python3-pip apt package and pip3 install -U).
while command -v pip3 >/dev/null 2>&1; do
    rm -f "$(command -v pip3)"
done
while command -v pip >/dev/null 2>&1; do
    rm -f "$(command -v pip)"
done
