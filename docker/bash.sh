#!/usr/bin/env bash

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
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh [-i] <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, default non-interactive
#     With -i, execute interactively.
#
interactive=0
if [ "$1" == "-i" ]; then
    interactive=1
    shift
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh [-i] <CONTAINER_NAME> [COMMAND]"
    exit -1
fi

DOCKER_IMAGE_NAME=("$1")

CI_DOCKER_EXTRA_PARAMS=( )
if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    interactive=1
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        CI_DOCKER_EXTRA_PARAMS=( "${CI_DOCKER_EXTRA_PARAMS[@]}" "-p 8888:8888" )
    else
        CI_DOCKER_EXTRA_PARAMS=( "${CI_DOCKER_EXTRA_PARAMS[@]}" "--net=host" )
    fi
else
    shift 1
    COMMAND=("$@")
fi

if [ $interactive -eq 1 ]; then
    CI_DOCKER_EXTRA_PARAMS=( "${CI_DOCKER_EXTRA_PARAMS[@]}" -it )
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

# Use nvidia-docker if the container is GPU.
if [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_ENV="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    CUDA_ENV=""
fi

if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* ]]; then
    if ! type "nvidia-docker" 1> /dev/null 2> /dev/null
    then
        DOCKER_BINARY="docker"
        CUDA_ENV=" --gpus all "${CUDA_ENV}
    else
        DOCKER_BINARY="nvidia-docker"
    fi
else
    DOCKER_BINARY="docker"
fi

if [[ "${DOCKER_IMAGE_NAME}" == *"ci"* ]]; then
    CI_PY_ENV="-e PYTHONPATH=/workspace/python"
else
    CI_PY_ENV=""
fi

# If the Vitis-AI docker image is selected, expose the Xilinx FPGA devices and required volumes containing e.g. DSA's and overlays
if [[ "${DOCKER_IMAGE_NAME}" == *"demo_vitis_ai"* && -d "/dev/shm" && -d "/opt/xilinx/dsa" && -d "/opt/xilinx/overlaybins" ]]; then
    WORKSPACE_VOLUMES="-v /dev/shm:/dev/shm -v /opt/xilinx/dsa:/opt/xilinx/dsa -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins"
    XCLMGMT_DRIVER="$(find /dev -name xclmgmt\*)"
    DOCKER_DEVICES=""
    for i in ${XCLMGMT_DRIVER} ;
    do
       DOCKER_DEVICES+="--device=$i "
    done

    RENDER_DRIVER="$(find /dev/dri -name renderD\*)"
    for i in ${RENDER_DRIVER} ;
    do
        DOCKER_DEVICES+="--device=$i "
    done

else
    DOCKER_DEVICES=""
    WORKSPACE_VOLUMES=""
fi


# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# When running from a git worktree, also mount the original git dir.
EXTRA_MOUNTS=( )
if [ -f "${WORKSPACE}/.git" ]; then
    git_dir=$(cd ${WORKSPACE} && git rev-parse --git-common-dir)
    if [ "${git_dir}" != "${WORKSPACE}/.git" ]; then
        EXTRA_MOUNTS=( "${EXTRA_MOUNTS[@]}" -v "${git_dir}:${git_dir}" )
    fi
fi

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
${DOCKER_BINARY} run --rm --pid=host\
    ${DOCKER_DEVICES}\
    ${WORKSPACE_VOLUMES}\
    -v ${WORKSPACE}:/workspace \
    -v ${SCRIPT_DIR}:/docker \
    "${EXTRA_MOUNTS[@]}" \
    -w /workspace \
    --ulimit stack=16777216:16777216 \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "CI_PYTEST_ADD_OPTIONS=$CI_PYTEST_ADD_OPTIONS" \
    ${CI_PY_ENV} \
    ${CUDA_ENV} \
    "${CI_DOCKER_EXTRA_PARAMS[@]}" \
    ${DOCKER_IMAGE_NAME} \
    bash --login /docker/with_the_same_user \
    ${COMMAND[@]}
