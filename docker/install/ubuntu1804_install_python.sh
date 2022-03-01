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
apt-get install -y software-properties-common
apt-get install -y python3.7 python3.7-dev python3-pip
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Pin pip and setuptools versions
# Hashes generated via:
#   $ pip download <package>==<version>
#   $ pip hash --algorithm sha512 <package>.whl
cat <<EOF > base-requirements.txt
pip==19.3.1 --hash=sha256:6917c65fc3769ecdc61405d3dfd97afdedd75808d200b2838d7d961cebc0c2c7
setuptools==58.4.0 --hash=sha256:e8b1d3127a0441fb99a130bcc3c2bf256c2d3ead3aba8fd400e5cbbaf788e036
EOF
pip3 install -r base-requirements.txt
