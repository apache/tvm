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

# We need to fix the onnx version because changing versions tends to break tests
# TODO(mbrookhart): periodically update

# onnx 1.9 removed onnx optimizer from the main repo (see
# https://github.com/onnx/onnx/pull/2834).  When updating the CI image
# to onnx>=1.9, onnxoptimizer should also be installed.
pip3 install \
    onnx==1.12.0 \
    onnxruntime==1.12.1 \
    onnxoptimizer==0.2.7

# torch depends on a number of other packages, but unhelpfully, does
# not expose that in the wheel!!!
pip3 install future

pip3 install \
    torch==1.12.0 \
    torchvision==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
