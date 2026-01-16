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


cleanup() {
  rm -rf base-requirements.txt
}

trap cleanup 0

# Install python and pip. Don't modify this to add Python package dependencies,
# instead modify install_python_package.sh
apt-get update
apt-install-and-clear -y python3.10 python3.10-dev python3-pip
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Pin pip and setuptools versions
# Hashes generated via:
#   $ pip download <package>==<version>
#   $ pip hash --algorithm sha256 <package>.whl
cat <<EOF > base-requirements.txt
pip==25.3 ; python_version >= "3.10" and python_version < "4.0" \
    --hash=sha256:8d0538dbbd7babbd207f261ed969c65de439f6bc9e5dbd3b3b9a77f25d95f343 \
    --hash=sha256:9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd
setuptools==80.9.0 ; python_version >= "3.10" and python_version < "4.0" \
    --hash=sha256:062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922 \
    --hash=sha256:f36b47402ecde768dbfafc46e8e4207b4360c654f1f3bb84475f0a28628fb19c
EOF
pip3 install -r base-requirements.txt
