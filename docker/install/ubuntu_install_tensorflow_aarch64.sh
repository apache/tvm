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

set -euxo pipefail

# Build dependencies
apt-install-and-clear -y --no-install-recommends libhdf5-dev

# Downloading Tensorflow and installing it manually is needed
# just as a temporary workaround while we move to a newer
# version (>2.7) that is hosted in the official PyPI repository.
linaro_repo="https://snapshots.linaro.org/ldcg/python/tensorflow-manylinux/43/tensorflow-aarch64"
tensorflow_package="tensorflow_aarch64-2.6.3-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

cd "${tmpdir}"
wget -q "${linaro_repo}/${tensorflow_package}"

# We're only using the TensorFlow wheel snapshot here as the
# h5py wheel tries to use the wrong .so file
pip3 install \
    ${tensorflow_package} \
    "h5py==3.1.0" \
    keras==2.6 \
    "protobuf<4"
