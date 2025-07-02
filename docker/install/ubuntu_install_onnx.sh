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

# Get the Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Set default value for first argument
DEVICE=${1:-cpu}

# Install the onnx package
pip3 install future

if [ "$PYTHON_VERSION" == "3.9" ]; then
    pip3 install \
        onnx==1.16.0 \
        onnxruntime==1.19.2 \
        onnxoptimizer==0.2.7

    if [ "$DEVICE" == "cuda" ]; then
        pip3 install \
            torch==2.7.0 \
            torchvision==0.22.0 \
            --index-url https://download.pytorch.org/whl/cu118
    else
        pip3 install \
            torch==2.7.0 \
            torchvision==0.22.0 \
            --extra-index-url https://download.pytorch.org/whl/cpu
    fi
elif [ "$PYTHON_VERSION" == "3.11" ]; then
    pip3 install \
        onnx==1.17.0 \
        onnxruntime==1.20.1 \
        onnxoptimizer==0.2.7

    if [ "$DEVICE" == "cuda" ]; then
        pip3 install \
            torch==2.7.0 \
            torchvision==0.22.0 \
            --index-url https://download.pytorch.org/whl/cu118
    else
        pip3 install \
            torch==2.7.0 \
            torchvision==0.22.0 \
            --extra-index-url https://download.pytorch.org/whl/cpu
    fi
else
    pip3 install \
        onnx==1.12.0 \
        onnxruntime==1.12.1 \
        onnxoptimizer==0.2.7

    if [ "$DEVICE" == "cuda" ]; then
        pip3 install \
            torch==2.4.1 \
            torchvision==0.19.1
    else
        pip3 install \
            torch==2.4.1 \
            torchvision==0.19.1 \
            --extra-index-url https://download.pytorch.org/whl/cpu
    fi
fi
