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

# Directories
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
driver_dir="${script_dir}/build/ethosu_core_driver/"
arm_dir="/opt/arm/"

# Make build directory
mkdir -p build
cd build

# Get mobilenet_v1 tflite model
mobilenet_url='https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz'
curl --retry 64 -sSL ${mobilenet_url} | gunzip | tar -xvf - ./mobilenet_v1_1.0_224_quant.tflite

# Compile model for Arm(R) Cortex(R)-M55 CPU and Ethos(TM)-U55 NPU
tvmc compile --target="ethos-u -accelerator_config=ethos-u55-256, \
    c -runtime=c --link-params -mcpu=cortex-m55 --executor=aot --interface-api=c --unpacked-api=1" \
    --pass-config tir.disable_vectorize=1 ./mobilenet_v1_1.0_224_quant.tflite --output-format=mlf
tar -xvf module.tar

# Get ImageNet labels
curl -sSL  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt \
    > ./labels_mobilenet_quant_v1_224.txt

# Get input image
curl -sSL https://s3.amazonaws.com/model-server/inputs/kitten.jpg > ./kitten.jpg

# Create C header files
cd ..
python3 ./convert_image.py ./build/kitten.jpg
python3 ./convert_labels.py ./build/labels_mobilenet_quant_v1_224.txt

# Build demo executable
cd ${script_dir}
make

# Run demo executable on the FVP
${arm_dir}/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -C cpu0.CFGDTCMSZ=15 \
-C cpu0.CFGITCMSZ=15 -C mps3_board.uart0.out_file=\"-\" -C mps3_board.uart0.shutdown_tag=\"EXITTHESIM\" \
-C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 \
-C mps3_board.telnetterminal1.start_telnet=0 -C mps3_board.telnetterminal2.start_telnet=0 -C mps3_board.telnetterminal5.start_telnet=0 \
-C ethosu.extra_args="--fast" \
-C ethosu.num_macs=256 ./build/demo
