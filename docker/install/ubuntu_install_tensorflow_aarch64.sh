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
apt-get install -y --no-install-recommends libhdf5-dev

# We're only using the TensorFlow wheel snapshot here as the
# h5py wheel tries to use the wrong .so file
pip3 install \
    "h5py==3.1.0" \
    keras==2.6 \
    tensorflow-aarch64==2.6.2 \
    -f https://snapshots.linaro.org/ldcg/python-cache/tensorflow-aarch64/
