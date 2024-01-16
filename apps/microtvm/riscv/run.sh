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

# Predefined models
declare -A MODELS=( \
    ['DS_CNN_S']='https://github.com/ARM-software/ML-examples/raw/9da709d96e5448520521e17165637c056c9bfae7/tflu-kws-cortex-m/Pretrained_models/DS_CNN/DS_CNN_S/ds_cnn_s_quantized.tflite' \
    ['DS_CNN_M']='https://github.com/ARM-software/ML-examples/raw/9da709d96e5448520521e17165637c056c9bfae7/tflu-kws-cortex-m/Pretrained_models/DS_CNN/DS_CNN_M/ds_cnn_m_quantized.tflite' \
    ['KWS_MICRONET_M']='https://github.com/ARM-software/ML-zoo/raw/aec70becff8d916ddec66efe3d02beba21fe8aad/models/keyword_spotting/micronet_medium/tflite_int8/kws_micronet_m.tflite' \
    ['PERSON_DETECT']='https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite' \
)

(( $# != 0 )) && [ -n "${MODELS[$1]+1}" ] || {
    echo "Wrong model name! Could be one of:"
    for key in "${!MODELS[@]}"; do echo "  ${key}"; done
}

MODEL=$1
MODEL_URL="${MODELS[${MODEL}]}"
MODEL_FILE='model.tflite'
MODULE_NAME='model'
MODEL_INPUT=

# Directories
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make build directory
make cleanall
mkdir -p build
cd build

# Get model
curl --retry 64 -sSL ${MODEL_URL} -o ./${MODEL_FILE}

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
    ./${MODEL_FILE}
tar -xf module.tar

# Create C header files and defines program options
cd ..
case ${MODEL} in
  DS_CNN_*)
    python3 ./convert_data.py ./data/yes.npy -t float32 -o 12
    export MODEL_FLAGS=" \
        -D MODEL_NAME=${MODEL} \
        -D MODEL_OUT_FIELD=.Identity \
        -D MODEL_IN_FIELD=.input \
        -D MODEL_OUT_TYPE=float \
        -D MODEL_VALID_RES=2 \
        -D MODEL_VALID_NAME=yes"
    ;;
  KWS_MICRONET_*)
    curl -sSL https://github.com/ARM-software/ML-zoo/raw/aec70becff8d916ddec66efe3d02beba21fe8aad/models/keyword_spotting/micronet_medium/tflite_int8/testing_input/input/0.npy -o ./build/0.npy
    python3 ./convert_data.py ./build/0.npy -t int8 -o 12
    export MODEL_FLAGS=" \
        -D MODEL_NAME=${MODEL} \
        -D MODEL_OUT_FIELD=.Identity \
        -D MODEL_IN_FIELD=.input \
        -D MODEL_OUT_TYPE=int8_t \
        -D MODEL_VALID_RES=11 \
        -D MODEL_VALID_NAME=go"
    ;;
  PERSON_DETECT)
    curl -sS https://raw.githubusercontent.com/tensorflow/tflite-micro/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp -o ./build/input_image.bmp
    python3 ./convert_data.py ./build/input_image.bmp -t int8 -o 2 -W 224 -H 224
    export MODEL_FLAGS=" \
        -D MODEL_NAME=${MODEL} \
        -D MODEL_OUT_FIELD=.MobilenetV1_Predictions_Reshape_1 \
        -D MODEL_IN_FIELD=.input \
        -D MODEL_OUT_TYPE=int8_t \
        -D MODEL_VALID_RES=1 \
        -D MODEL_VALID_NAME=person"
    ;;
esac

# Build executable
cd ${script_dir}

(( $# > 1 )) && [ "$2" == "32" ] && IS_32_BIT=1
(( $# > 2 )) && [ "$3" == "spike" ] && IS_SPIKE=1

if [ -z ${IS_32_BIT+x} ]; then
  MARCH=rv64gc
  MABI=lp64d
  PK=pk64
  SIM=qemu-riscv64
else
  MARCH=rv32imac
  MABI=ilp32
  PK=pk
  SIM=qemu-riscv32
fi

TOOLCHAIN_PREFIX="riscv64-unknown-elf-"

if [ ! -z ${IS_SPIKE+x} ]; then
  SIM="spike --isa=${MARCH} $(${TOOLCHAIN_PREFIX}gcc -print-sysroot)/bin/${PK}"
fi

make TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX} \
     COMPILE_OPTS="-march=${MARCH} -mabi=${MABI}" \
     LDFLAGS="-march=${MARCH} -mabi=${MABI}"

${SIM} ./build/${MODULE_NAME}
