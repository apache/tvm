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

# Install jax and jaxlib
if [ "$1" == "cuda" ]; then
    pip3 install --upgrade \
        jaxlib~=0.4.9 \
        "jax[cuda11_pip]~=0.4.9" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    pip3 install --upgrade \
        jaxlib~=0.4.9 \
        "jax[cpu]~=0.4.9"
fi

# Install flax
pip3 install flax~=0.6.9
