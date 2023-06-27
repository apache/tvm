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
Usage: run_demo.sh [--ethosu_driver_path ETHOSU_DRIVER_PATH] [--alif_toolkit_path ALIF_TOOLKIT_PATH]
-h, --help
    Display this help message.
--ethosu_driver_path ETHOSU_DRIVER_PATH
    Set path to Arm(R) Ethos(TM)-U core driver.
--alif_toolkit_path
   Set path to Alif's toolkit.
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

        --ethosu_driver_path)
            if [ $# -gt 1 ]
            then
                export ETHOSU_DRIVER_PATH="$2"
                shift 2
            else
                echo 'ERROR: --ethosu_driver_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
            ;;

        --alif_toolkit_path)
            if [ $# -gt 1 ]
            then
                export ALIF_TOOLKIT_PATH="$2"
                shift 2
            else
                echo 'ERROR: --alif_toolkit_path requires a non-empty argument' >&2
                show_usage >&2
                exit 1
            fi
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

        --freertos_path)
            if [ $# -gt 1 ]
            then
                export FREERTOS_PATH="$2"
                shift 2
            else
                echo 'ERROR: --freertos_path requires a non-empty argument' >&2
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

make cleanall

if [ -n "${ALIF_TOOLKIT_PATH+x}" ]; then
    mkdir -p ${script_dir}/build/dependencies
    cd ${script_dir}/build/dependencies

    # Clone Alif's evaluation kit
    ALIF_ML_KIT_VERSION="07b186198f148994683d970d57983bfe0eb996bb"
    wget -q https://github.com/alifsemi/alif_ml-embedded-evaluation-kit/archive/${ALIF_ML_KIT_VERSION}.zip -O ALIF_Ml_KIT.zip
    unzip -q ALIF_Ml_KIT.zip -d .
    mv alif_ml-embedded-evaluation-kit-${ALIF_ML_KIT_VERSION} alif_ml-embedded-evaluation-kit
    rm ALIF_Ml_KIT.zip

    ALIF_CMSIS_VERSION="833ffaba7ddeb3b59e2786a7acab215a62a0b617"
    wget -q https://github.com/alifsemi/alif_ensemble-cmsis-dfp/archive/${ALIF_CMSIS_VERSION}.zip -O ALIF_CMSIS.zip
    unzip -q ALIF_CMSIS.zip -d .
    mv alif_ensemble-cmsis-dfp-${ALIF_CMSIS_VERSION} alif_ensemble-cmsis-dfp
    rm ALIF_CMSIS.zip
fi

# Make build directory
mkdir -p ${script_dir}/build
cd ${script_dir}/build

# Get mobilenet_v2 tflite model
mobilenet_url='https://github.com/ARM-software/ML-zoo/raw/b9e26e662c00e0c0b23587888e75ac1205a99b6e/models/image_classification/mobilenet_v2_1.0_224/tflite_int8/mobilenet_v2_1.0_224_INT8.tflite'
curl --retry 64 -sSL ${mobilenet_url} -o ./mobilenet_v2_1.0_224_INT8.tflite

# Compile model for Arm(R) Cortex(R)-M55 CPU and Ethos(TM)-U55 NPU
# An alternative to using "python3 -m tvm.driver.tvmc" is to call
# "tvmc" directly once TVM has been pip installed.
python3 -m tvm.driver.tvmc compile --target=ethos-u,cmsis-nn,c \
    --target-ethos-u-accelerator_config=ethos-u55-256 \
    --target-cmsis-nn-mcpu=cortex-m55 \
    --target-c-mcpu=cortex-m55 \
    --runtime=crt \
    --executor=aot \
    --executor-aot-interface-api=c \
    --executor-aot-unpacked-api=1 \
    --pass-config tir.usmp.enable=1 \
    --pass-config tir.usmp.algorithm=hill_climb \
    --pass-config tir.disable_storage_rewrite=1 \
    --pass-config tir.disable_vectorize=1 ./mobilenet_v2_1.0_224_INT8.tflite --output-format=mlf
tar -xf module.tar

# Get ImageNet labels
curl -sS  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt \
    -o ./labels_mobilenet_quant_v1_224.txt

# Get input image
curl -sS https://s3.amazonaws.com/model-server/inputs/kitten.jpg -o kitten.jpg

# Create C header files
cd ${script_dir}
python3 ./convert_image.py ./build/kitten.jpg
python3 ./convert_labels.py ./build/labels_mobilenet_quant_v1_224.txt

if [ -n "${ALIF_TOOLKIT_PATH+x}" ]; then
    # Build alif demo executable
    make -f Makefile_alif.mk demo_alif
    echo 

else
    # Build demo executable
    make

    # Run demo executable on the FVP
    FVP_Corstone_SSE-300_Ethos-U55 -C cpu0.CFGDTCMSZ=15 \
    -C cpu0.CFGITCMSZ=15 -C mps3_board.uart0.out_file=\"-\" -C mps3_board.uart0.shutdown_tag=\"EXITTHESIM\" \
    -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.telnetterminal1.start_telnet=0 -C mps3_board.telnetterminal2.start_telnet=0 -C mps3_board.telnetterminal5.start_telnet=0 \
    -C ethosu.extra_args="--fast" \
    -C ethosu.num_macs=256 ./build/demo
fi
