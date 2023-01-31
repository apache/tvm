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

function show_usage() {
    cat <<EOF
This script is for running microtvm_api_server with Zephyr.
Usage: launch_microtvm_api_server.sh <microtvm_api_server.py> --read-fd <READ_FD_PATH> --write-fd <WRITE_FD_PATH>
EOF
}

if [ "$#" -lt 5 -o "$1" == "--help" ]; then
    show_usage
    exit -1
fi

west_file_path=$(which west)

# Remove space and extra characters
line=$(head -n 1 ${west_file_path})
line=$(echo ${line} | sed 's/ //')
line=$(echo ${line} | sed 's/!//')
line=$(echo ${line} | sed 's/#//')
PYTHON_CMD=$line

# Run server
$PYTHON_CMD $1 $2 $3 $4 $5
