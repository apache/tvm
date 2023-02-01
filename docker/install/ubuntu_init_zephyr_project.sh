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

#
# Initialize Zephyr Project.
#
# Usage: docker/install/ubuntu_init_zephyr_project.sh <INSTALLATION_PATH> [--branch BRANCH]
#         [--commit HASH]
# INSTALLATION_PATH is the installation path for the repository.
# --branch is the zephyr branch. If not specified, it uses the default.
# --commit is the commit hash number of zephyrproject repository. If not specified, it uses the latest commit.
#

set -x

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_init_zephyr_project.sh <INSTALLATION_PATH> [--branch BRANCH]
        [--commit COMMIT]
INSTALLATION_PATH is the installation path for the repository.
--branch is the zephyr branch. If not specified, it uses the default.
--commit is the commit hash number of zephyrproject repository. If not specified, it uses the latest commit.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit -1
fi

INSTALLATION_PATH=$1
shift

if [ "$1" == "--branch" ]; then
    shift
    BRANCH=$1
    shift
else
    BRANCH="v3.2-branch"
fi

COMMIT=
if [ "$1" == "--commit" ]; then
    shift
    COMMIT=$1
    shift
fi

west init --mr ${BRANCH} ${INSTALLATION_PATH}

if [ -n "$COMMIT" ]; then
    cd ${INSTALLATION_PATH}/zephyr
    git checkout ${COMMIT}
fi

cd ${INSTALLATION_PATH}
west update
west zephyr-export
