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
set -x
set -o pipefail

MODEL_URL=https://github.com/ARM-software/ML-zoo/raw/ee35139af86bdace5e502b09fe8b9da9cb1f06bb/models/keyword_spotting/cnn_small/tflite_int8
MODEL_FILE=cnn_s_quantized.tflite

LOGDIR="$(mktemp -d)"

cleanup()
{
  rm -rf "$LOGDIR"
  pkill FVP
}

trap cleanup EXIT

# Clean up previous build
rm -rf build

# Download model file
wget $MODEL_URL/$MODEL_FILE -O model/$MODEL_FILE

# System doesn't automatically exit so we wait for the output
# and kill it ourselves
export ARMFVP_BIN_PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/
west zephyr-export
west build
west build -t run &> ${LOGDIR}/west.log &

# Wait for "exit" keyword
until grep -m 1 "exit" ${LOGDIR}/west.log; do sleep 1 ; done

# Check the log for correct output
grep "The word is 'down'!" ${LOGDIR}/west.log
