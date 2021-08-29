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


function filter_jenkinsfile() {
    local echo_on=0;
    while read line; do
        if [ "${line}" == "// NOTE: these lines are scanned by docker/dev_common.sh. Please update the regex as needed. -->" ]; then
            echo_on=1
        elif [ "${line}" == "// <--- End of regex-scanned config." ]; then
            break
        elif [ ${echo_on} -eq 1 ]; then
            echo "$line"
        fi
    done
}


function lookup_image_spec() {
    img_line=$(cat "${GIT_TOPLEVEL}/Jenkinsfile" | filter_jenkinsfile | grep -E "^${1} = ")
    if [ -n "${img_line}" ]; then
        img_spec=$(echo "${img_line}" | sed -E "s/${1} = \"([^\"]*)\"/\1/")
        has_similar_docker_image=1
        docker inspect "${1}" &>/dev/null || has_similar_docker_image=0
        if [ ${has_similar_docker_image} -ne 0 ]; then
            echo "WARNING: resolved docker image through Jenkinsfile to \"${img_spec}\"" >&2
        fi
        echo "${img_spec}"
    fi
}


function run_docker() {
    image_name="$1"  # Name of the Jenkinsfile var to find
    shift

    image_spec=$(lookup_image_spec "${image_name}")
    if [ -z "${image_spec}" ]; then
        echo "${image_name}: not found in ${GIT_TOPLEVEL}/Jenkinsfile" >&2
        exit 2
    fi

    "${GIT_TOPLEVEL}/docker/bash.sh" -i "${image_spec}" "$@"
}
