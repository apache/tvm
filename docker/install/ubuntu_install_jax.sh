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

JAX_VERSION=0.4.30

# Install jaxlib
if [ "$1" == "cuda" ]; then
    pip install -U \
        "jax[cuda12]~=${JAX_VERSION}" \
        jaxlib~=${JAX_VERSION}
else
    pip3 install -U \
        jax~=${JAX_VERSION} \
        jaxlib~=${JAX_VERSION}
fi

# Install flax
pip3 install flax~=0.8.5
