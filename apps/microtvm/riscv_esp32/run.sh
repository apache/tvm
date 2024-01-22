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

declare -A MODELS=( \
    ['DS_CNN_S']='https://github.com/ARM-software/ML-examples/raw/9da709d96e5448520521e17165637c056c9bfae7/tflu-kws-cortex-m/Pretrained_models/DS_CNN/DS_CNN_S/ds_cnn_s_quantized.tflite' \
    ['DS_CNN_M']='https://github.com/ARM-software/ML-examples/raw/9da709d96e5448520521e17165637c056c9bfae7/tflu-kws-cortex-m/Pretrained_models/DS_CNN/DS_CNN_M/ds_cnn_m_quantized.tflite' \
)

MODEL_URL="${MODELS[DS_CNN_S]}"

MODULE_NAME='kws'

# Directories
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make build directory
BUILD_DIR=${PWD}/build
mkdir -p ${BUILD_DIR}

# Get lib for MFCC calculation
NMSIS_TAG=1.2.0
NMSIS_DIR=NMSIS-${NMSIS_TAG}
if [ ! -d "${BUILD_DIR}/${NMSIS_DIR}" ]; then
  curl --retry 64 -sSL https://github.com/Nuclei-Software/NMSIS/archive/refs/tags/${NMSIS_TAG}.tar.gz | tar -xzf - -C ${BUILD_DIR}
fi

# Get model
MODEL_FILE=${BUILD_DIR}/model.tflite
curl --retry 64 -sSL ${MODEL_URL} -o ${MODEL_FILE}

# Compile model
python3 -m tvm.driver.tvmc compile --target=c \
    --target-c-keys=riscv_cpu \
    --runtime=crt \
    --executor=aot \
    --executor-aot-interface-api=c \
    --executor-aot-unpacked-api=1 \
    --pass-config tir.disable_vectorize=1 \
    --output-format=mlf \
    --module-name=${MODULE_NAME} \
    -o ${BUILD_DIR}/module.tar \
    ${MODEL_FILE}
tar -xf ${BUILD_DIR}/module.tar -C ${BUILD_DIR}

# Create C header files
python3 allocate_tensors.py ${MODEL_FILE} -B ${BUILD_DIR}

# Set target
export IDF_TARGET=esp32c3

# Build executable
idf.py build -D WITH_TVM_MODULE=${MODULE_NAME} -D NMSIS=${BUILD_DIR}/${NMSIS_DIR}/NMSIS

# Flash and run monitor
idf.py flash
idf.py monitor
