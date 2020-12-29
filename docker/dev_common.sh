#!/bin/bash -e
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


if [ -z "${BASH_SOURCE[0]}" ]; then
    echo "NOTE: This script must be source'd from another bash script--it cannot be run directly"
    exit 2
fi

INVOCATION_PWD="$(pwd)"


GIT_TOPLEVEL=$(cd $(dirname ${BASH_SOURCE[0]}) && git rev-parse --show-toplevel)


function run_docker() {
    image_name="$1"  # Name of the Jenkinsfile var to find
    shift

    image_spec=$(cat "${GIT_TOPLEVEL}/Jenkinsfile" | \
                     grep -E "^${image_name} = " | \
                     sed -E "s/${image_name} = \"([^\"]*)\"/\1/")
    if [ -z "${image_spec}" ]; then
        echo "${image_name}: not found in ${GIT_TOPLEVEL}/Jenkinsfile" >&2
        exit 2
    fi

    "${GIT_TOPLEVEL}/docker/bash.sh" -i "${image_spec}" "$@"
}
