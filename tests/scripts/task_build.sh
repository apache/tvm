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
set -eux

VTA_HW_PATH=$(pwd)/3rdparty/vta-hw
export VTA_HW_PATH

pushd "$1"

# Set up sccache to use S3
if [ -n "${3+x}" ] && which sccache; then
    export SCCACHE_BUCKET=$3
    HAS_SCCACHE=1
else
    export SCCACHE_BUCKET="no-bucket-configured"
    HAS_SCCACHE=0
fi

if [ "$HAS_SCCACHE" -eq "1" ]; then
    echo "Running with sccache enabled"
    export CC=/opt/sccache/cc
    export CXX=/opt/sccache/c++
fi

# Send this out through a side channel, the bucket name is not actually secret
# so it's OK to leak it in this way
echo "$SCCACHE_BUCKET" | base64

if [ "$HAS_SCCACHE" -eq "1" ]; then
    sccache --start-server || echo failed
    echo "===== sccache stats ====="
    sccache --show-stats
fi

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . -- VERBOSE=1 "$2"

if [ "$HAS_SCCACHE" -eq "1" ]; then
    echo "===== sccache stats ====="
    sccache --show-stats
fi
popd
