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
set -x

# Show usage
function show_usage() {
    cat <<EOF
Usage: run_demo.sh
-h, --help
    Display this help message.
--cmsis_path CMSIS_PATH
    Set path to CMSIS.
--ethosu_platform_path ETHOSU_PLATFORM_PATH
    Set path to Arm(R) Ethos(TM)-U core platform.
--fvp_path FVP_PATH
   Set path to FVP.
--cmake_path
   Set path to cmake.
EOF
}

# Parse arguments
while (( $# )); do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;

        --cmsis_path)
            if [ $# -gt 1 ]
            then
                export CMSIS_PATH="$2"
                shift 2
            else
                echo 'ERROR: --cmsis_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
            ;;

        --ethosu_platform_path)
            if [ $# -gt 1 ]
            then
                export ETHOSU_PLATFORM_PATH="$2"
                shift 2
            else
                echo 'ERROR: --ethosu_platform_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
            ;;

        --fvp_path)
            if [ $# -gt 1 ]
            then
                export PATH="$2/models/Linux64_GCC-6.4:$PATH"
                shift 2
            else
                echo 'ERROR: --fvp_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
            ;;

        --cmake_path)
            if [ $# -gt 1 ]
            then
                export CMAKE="$2"
                shift 2
            else
                echo 'ERROR: --cmake_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
            ;;

        -*|--*)
            echo "Error: Unknown flag: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done


# Directories
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make build directory
make cleanall
mkdir -p build
cd build

# Get person_detect model
model_url='https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/models/person_detect.tflite'
curl --retry 64 -sSL ${model_url} -o ./person_detect.tflite

# Compile model for Arm(R) Cortex(R)-M55 CPU and CMSIS-NN
# An alternative to using "python3 -m tvm.driver.tvmc" is to call
# "tvmc" directly once TVM has been pip installed.
python3 -m tvm.driver.tvmc compile --target=cmsis-nn,c \
    --target-cmsis-nn-mcpu=cortex-m55 \
    --target-c-mcpu=cortex-m55 \
    --runtime=crt \
    --executor=aot \
    --executor-aot-interface-api=c \
    --executor-aot-unpacked-api=1 \
    --pass-config tir.usmp.enable=1 \
    --pass-config tir.usmp.algorithm=hill_climb \
    --pass-config tir.disable_storage_rewrite=1 \
    --pass-config tir.disable_vectorize=1 ./person_detect.tflite \
    --output-format=mlf \
    --module-name=detection
tar -xf module.tar

# Get input image
curl -sS https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp -o input_image.bmp
# curl -sS https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/testdata/no_person.bmp -o input_image.bmp

# Create C header files
cd ..
python3 ./convert_image.py ./build/input_image.bmp

# Build demo executable
cd ${script_dir}
make

# Run demo executable on the FVP
FVP_Corstone_SSE-300_Ethos-U55 -C cpu0.CFGDTCMSZ=15 \
-C cpu0.CFGITCMSZ=15 -C mps3_board.uart0.out_file=\"-\" -C mps3_board.uart0.shutdown_tag=\"EXITTHESIM\" \
-C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 \
-C mps3_board.telnetterminal1.start_telnet=0 -C mps3_board.telnetterminal2.start_telnet=0 -C mps3_board.telnetterminal5.start_telnet=0 \
./build/demo
