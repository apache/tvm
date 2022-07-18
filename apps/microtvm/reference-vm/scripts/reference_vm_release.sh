#!/bin/bash -e
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
#

if [ "$#" -lt 2 -o "$1" == "--help" -o "$1" == "-h" ]; then
    echo "Usage: apps/microtvm/reference-vm/scripts/reference_vm_release.sh <RELEASE_NAME> <RELEASE_VERSION>"
    exit -1
fi

RELEASE_NAME=$1
shift

RELEASE_VERSION=$1
shift

cd "$(dirname "$0")"
source "./utils.sh" || exit 2
cd ${RVM_BASE_PATH}

${BASE_BOX_TOOL} --provider=virtualbox release \
    --release-full-name=${RELEASE_NAME} \
    --release-version=${RELEASE_VERSION} \
    --skip-creating-release-version
