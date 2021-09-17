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
# Usage: ubuntu_init_zephyr_project.sh path branch [--commit hash]
#   path is the installation path for the repository.
#   branch is the zephyr branch.
#   --commit is the commit hash number of zephyrproject repository. If not specified, it uses the latest commit.
#

set -x

DOWNLOAD_DIR=$1
shift
ZEPHYR_BRANCH=$1
shift

commit_hash=
if [ "$1" == "--commit" ]; then
    shift
    commit_hash=$1
fi

west init --mr ${ZEPHYR_BRANCH} ${DOWNLOAD_DIR}

if [ -n "$commit_hash" ]; then
    cd ${DOWNLOAD_DIR}/zephyr
    git checkout ${commit_hash}
fi

cd ${DOWNLOAD_DIR}
west update
west zephyr-export
