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
pip==22.0.3 --hash=sha512:12ca75130a1ce9807060a66dd2341afc7c7e663357ca4b937868dbc733634e11bae49ffff96acb0f5f3fb16cb14f680b9b6d185155a711c6098eda5cfbf2f8f5
setuptools==60.9.1 --hash=sha512:24c21006f0650209e6a934a5366614e32a98fbdf11bc0e941419731aa3c8cb6d217e486608b4834b8f89c14dc36b3bafa30e795c87b58705be2b156f78acc3c5
EOF
pip3 install -r base-requirements.txt
