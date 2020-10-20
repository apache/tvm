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

SKIP_ADD=( )
if [ "$1" == "--skip-add" ]; then
    SKIP_ADD=( "-var" "skip_add=true" )
    shift
fi

if [ $# -ne 1 ]; then
    echo "usage: $0 [--skip-add] <version>"
    exit 2
fi

cd "$(dirname $0)"
if [ ! -e api-token ]; then
    echo "must create a file named 'api-token' in $(pwd)"
    echo "file contents:"
    echo api_token = "<VAGRANT_CLOUD_API_TOKEN>"
    exit 2
fi

if [[ ! "$1" =~ ^[0-9].[0-9].[0-9]$ ]]; then
    echo "version: must be x.y.z; got $1"
    exit 2
fi

cat packer.json.template | sed 's/^#.*//g' >packer.json

ALL_PROVIDERS=( \
#    virtualbox \
    parallels \
)
for provider in "${ALL_PROVIDERS[@]}"; do
    set -x
    packer build -var-file=api-token "${SKIP_ADD[@]}" -var "provider=${provider}" -var "version=$1" packer.json
    set +x
done
