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
set -o pipefail

# Show usage
function show_usage() {
    cat <<EOF
Usage: flash.sh <--alif_toolkit_path path_to_SETools> <--alif_console_port tty_device>
-h, --help
    Display this help message.
--alif_toolkit_path
    Set path to the Alif SETools.
--alif_console_port
    Set Alif Evaluation Kit console port.
EOF
}

# Parse arguments
while (( $# )); do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
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

        --alif_console_port)
            if [ $# -gt 1 ]
            then
                export ALIF_CONSOLE_PORT="$2"
                shift 2
            else
                echo 'ERROR: --alif_console_port requires a non-empty argument' >&2
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

if [ -z "$ALIF_TOOLKIT_PATH" ] && [ -z "$ALIF_CONSOLE_PORT" ]; then
        echo 'Missing required argument' >&2
        show_usage >&2
        exit 1
fi

# Directories
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# copy demo artifacts to the Alis Toolkit folder
cp ${script_dir}/build/alif_flash_config.json ${ALIF_TOOLKIT_PATH}/build/config
cp ${script_dir}/build/demo_alif.bin ${ALIF_TOOLKIT_PATH}/build/images

# upload binary to the MCU
cd ${ALIF_TOOLKIT_PATH}
./app-gen-toc -f ./build/config/alif_flash_config.json
./app-write-mram

# Read the board's console output
stty -F ${ALIF_CONSOLE_PORT} 115200
stty -F ${ALIF_CONSOLE_PORT} time 10
cat ${ALIF_CONSOLE_PORT}
