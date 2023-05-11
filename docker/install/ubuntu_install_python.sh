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

if [ "${PYTHON_VERSION}" != "3.7" ] && [ "${PYTHON_VERSION}" != "3.8" ]; then
    echo "Only 3.7 and 3.8 versions are supported in this script."
    exit -1
fi

apt-get update

# Ensure lsb-release is installed.
apt-install-and-clear -y \
    lsb-core

apt-install-and-clear -y software-properties-common

release=$(lsb_release -sc)
if [ "${release}" == "bionic" ]; then
    if [ "${PYTHON_VERSION}" == "3.8" ]; then
        add-apt-repository -y ppa:deadsnakes/ppa
    fi
elif [ "${release}" == "focal" ]; then
    if [ "${PYTHON_VERSION}" == "3.7" ]; then
        add-apt-repository -y ppa:deadsnakes/ppa
    fi
elif [ "${release}" == "jammy" ]; then
    if [ "${PYTHON_VERSION}" == "3.8" ]; then
        add-apt-repository -y ppa:deadsnakes/ppa
    fi
else
    echo "Don't know which version of python to install for lsb-release ${release}"
    exit 2
fi

# Install python and pip. Don't modify this to add Python package dependencies,
# instead modify install_python_package.sh
apt-install-and-clear -y \
    acl \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python${PYTHON_VERSION}-venv

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Allow disabling user site-packages, even with sudo; this makes it harder to repro CI failures
# locally because it's hard to tell what might be in this directory.
echo "Defaults env_keep += \"PYTHONNOUSERSITE\"" >/etc/sudoers.d/91-preserve-python-nousersite
export PYTHONNOUSERSITE=1

venv_dir="$(python3 -c "import os.path;print(os.path.dirname(\"${TVM_VENV}\"))")"
mkdir -p "${venv_dir}"
python3 -mvenv "${TVM_VENV}"
. "${TVM_VENV}/bin/activate"

# NOTE: Only in python3.9 does venv guarantee it creates the python3.X binary.
# This is needed so that cmake's find_package(PythonInterp) works inside the venv.
# See https://bugs.python.org/issue39656
if [ ! -e "${TVM_VENV}/bin/python${PYTHON_VERSION}" ]; then
    ln -s "${TVM_VENV}/bin/python" "${TVM_VENV}/bin/python${PYTHON_VERSION}"
fi

# Update pip to match version used to produce requirements-hashed.txt. This step
# is necessary so that pip's dependency solver is recent.
pip_spec=$(cat /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt | grep 'pip==')
pip3 install -U --require-hashes -r <(echo "${pip_spec}") \
     -c /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt

# Python configuration
pip3 config set global.no-cache-dir true  # Never cache packages

# Now install the remaining base packages.
pip3 install \
     --require-hashes \
     -r /install/python/bootstrap/lockfiles/constraints-${PYTHON_VERSION}.txt

addgroup tvm-venv
chgrp -R tvm-venv "${TVM_VENV}"
setfacl -R -d -m group:tvm-venv:rwx "${TVM_VENV}"
setfacl -R -m group:tvm-venv:rwx "${TVM_VENV}"

# Prevent further use of pip3 via the system.
# There may be multiple (i.e. from python3-pip apt package and pip3 install -U).
deactivate
while [ "$(which pip3)" != "" ]; do
    rm "$(which pip3)"
done
while [ "$(which pip)" != "" ]; do
    rm "$(which pip)"
done
