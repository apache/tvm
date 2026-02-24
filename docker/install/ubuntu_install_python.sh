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

case "$PYTHON_VERSION" in
    3.10|3.11) ;;
    *)
        echo "Only 3.10 and 3.11 versions are supported in this script."
        exit -1
        ;;
esac

apt-get update

# Install base dependencies required by this script.
apt-install-and-clear -y acl

# Install managed Python to a shared location so both root and the CI runtime user can use it.
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/opt/uv/python}
export UV_PYTHON_INSTALL_DIR
mkdir -p "${UV_PYTHON_INSTALL_DIR}"
chmod 755 "${UV_PYTHON_INSTALL_DIR}"
uv python install "${PYTHON_VERSION}" --install-dir "${UV_PYTHON_INSTALL_DIR}" --no-bin

# Allow disabling user site-packages, even with sudo; this makes it harder to repro CI failures
# locally because it's hard to tell what might be in this directory.
echo "Defaults env_keep += \"PYTHONNOUSERSITE\"" >/etc/sudoers.d/91-preserve-python-nousersite
export PYTHONNOUSERSITE=1

venv_dir="$(dirname "${TVM_VENV}")"
mkdir -p "${venv_dir}"
uv venv "${TVM_VENV}" --python "${PYTHON_VERSION}" --managed-python --seed

# NOTE: Only in python3.9 does venv guarantee it creates the python3.X binary.
# This is needed so that CMake's find_package(PythonInterp) works inside the venv.
# See https://bugs.python.org/issue39656
if [ ! -e "${TVM_VENV}/bin/python${PYTHON_VERSION}" ]; then
    ln -s "${TVM_VENV}/bin/python" "${TVM_VENV}/bin/python${PYTHON_VERSION}"
fi

# Update pip to match version used to produce requirements-hashed.txt. This step
# is necessary so that pip's dependency solver is recent.
if [ -f "/install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt" ]; then
    pip_spec=$(tac /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt | grep -m 1 'pip==')
    ${TVM_VENV}/bin/pip install -U --require-hashes -r <(echo "${pip_spec}") \
        -c /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt

    # Python configuration
    ${TVM_VENV}/bin/pip config set global.no-cache-dir true  # Never cache packages

    # Now install the remaining base packages.
    ${TVM_VENV}/bin/pip install \
        --require-hashes \
        -r /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt
fi

addgroup tvm-venv
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
