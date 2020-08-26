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
# Usage: bash.sh <CONTAINER_TYPE> [-i] [--net=host] [--mount path] <CONTAINER_NAME>  <COMMAND>
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh [-i] <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, default non-interactive
#     With -i, execute interactively.
#

set -e

source "$(dirname $0)/dev_common.sh" || exit 2

interactive=0
if [ "$1" == "-i" ]; then
    interactive=1
    shift
fi

CI_DOCKER_EXTRA_PARAMS=( )
if [[ "$1" == "--net=host" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=('--net=host')
    shift 1
fi

# Mount external directory to the docker
CI_DOCKER_MOUNT_CMD=( )
if [ "$1" == "--mount" ]; then
    shift 1
    CI_DOCKER_MOUNT_CMD=( -v "$1:$1" )
    shift 1
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh [-i] [--net=host] <CONTAINER_NAME> [COMMAND]"
    exit -1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

if [ "$1" == "--repo-mount-point" ]; then
    shift
    REPO_MOUNT_POINT="$1"
    shift
else
    REPO_MOUNT_POINT="${WORKSPACE}"
fi

DOCKER_IMAGE_NAME=$(lookup_image_spec "$1")
if [ -z "${DOCKER_IMAGE_NAME}" ]; then
    DOCKER_IMAGE_NAME=("$1")
fi

if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    interactive=1
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        CI_DOCKER_EXTRA_PARAMS=( "${CI_DOCKER_EXTRA_PARAMS[@]}" "-p" "8888:8888" )
    else
        CI_DOCKER_EXTRA_PARAMS+=( "${CI_DOCKER_EXTRA_PARAMS[@]}" "--net=host" )
    fi
else
    shift 1
    COMMAND=("$@")
fi

if [ $interactive -eq 1 ]; then
    CI_DOCKER_EXTRA_PARAMS=( "${CI_DOCKER_EXTRA_PARAMS[@]}" -it )
fi

# Use nvidia-docker if the container is GPU.
if [[ ! -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_ENV="-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    CUDA_ENV=""
fi

if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* || "${DOCKER_IMAGE_NAME}" == *"cuda"* ]]; then
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
    CI_ADDON_ENV="-e PYTHONPATH=/workspace/python"
else
    CI_ADDON_ENV=""
fi

DOCKER_ENVS=""
DOCKER_DEVICES=""
WORKSPACE_VOLUMES=""
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
fi

# Add ROCm devices and set ROCM_ENABLED=1 which is used in the with_the_same_user script
# to add the user to the video group
if [[ "${DOCKER_IMAGE_NAME}" == *"rocm"* && -d "/dev/dri" ]]; then
    DOCKER_DEVICES+="--device=/dev/kfd --device=/dev/dri "
    DOCKER_ENVS+="-e ROCM_ENABLED=1 "
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
    -v ${WORKSPACE}:${REPO_MOUNT_POINT} \
    -v ${SCRIPT_DIR}:/docker \
    "${CI_DOCKER_MOUNT_CMD[@]}" \
    "${EXTRA_MOUNTS[@]}" \
    -w "${REPO_MOUNT_POINT}" \
    -e "CI_BUILD_HOME=${REPO_MOUNT_POINT}" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "CI_PYTEST_ADD_OPTIONS=$CI_PYTEST_ADD_OPTIONS" \
    -e "CI_IMAGE_NAME=${DOCKER_IMAGE_NAME}" \
    ${DOCKER_ENVS} \
    ${CI_ADDON_ENV} \
    ${CUDA_ENV} \
    "${CI_DOCKER_EXTRA_PARAMS[@]}" \
    ${DOCKER_IMAGE_NAME} \
    bash --login /docker/with_the_same_user \
    "${COMMAND[@]}"
