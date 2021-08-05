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

set -euo pipefail

function show_usage() {
    cat <<EOF
Usage: docker/bash.sh [-i|--interactive] [--net=host]
         [--mount MOUNT_DIR] [--dry-run]
         <DOCKER_IMAGE_NAME> [--] [COMMAND]

-h, --help

    Display this help message.

-i, --interactive

    Start the docker session in interactive mode.

--net=host

    Expose servers run into the container to the host, passing the
    "--net=host" argument through to docker.  On MacOS, this is
    instead passed as "-p 8888:8888" since the host networking driver
    isn't supported.

--mount MOUNT_DIR

    Expose MOUNT_DIR as an additional mount point inside the docker
    container.  The mount point inside the container is the same as
    the folder location outside the container.  This option can be
    specified multiple times.

--dry-run

    Print the docker command to be run, but do not execute it.

DOCKER_IMAGE_NAME

    The name of the docker container to be run.  This can be an
    explicit name of a docker image (e.g. "tlcpack/ci-gpu:v0.76") or
    can be a shortcut as defined in the TVM Jenkinsfile
    (e.g. "ci_gpu").

COMMAND

    The command to be run inside the docker container.  If this is set
    to "bash", both the --interactive and --net=host flags are set.
    If no command is specified, defaults to "bash".  If the command
    contains dash-prefixed arguments, the command should be preceded
    by -- to indicate arguments that are not intended for bash.sh.

EOF
}


#################################
### Start of argument parsing ###
#################################

DRY_RUN=false
INTERACTIVE=false
USE_NET_HOST=false
DOCKER_IMAGE_NAME=
COMMAND=bash
MOUNT_DIRS=( )

trap "show_usage >&2" ERR
args=$(getopt \
           --name bash.sh \
           --options "ih" \
           --longoptions "interactive,net=host,mount:,dry-run" \
           --longoptions "help" \
           --unquoted \
           -- "$@")
trap - ERR
set -- $args

while (( $# )); do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;

        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;

        --net=host)
            USE_NET_HOST=true
            shift
            ;;

        --mount)
            MOUNT_DIRS+=($2)
            shift
            shift
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;

        --)
            shift
            break
            ;;

        -*|--*)
            echo "Error: Unknown flag: $1" >&2
            echo "  If this flag is intended to be passed to the" >&2
            echo "  docker command, please add -- before the docker" >&2
            echo "  command (e.g. docker/bash.sh ci_gpu -- build -j2)" >&2
            show_usage >&2
            exit 1
            ;;

        *)
            echo "Internal Error: getopt should output -- before positional" >&2
            exit 2
            ;;
    esac
done

if (( $# )); then
    DOCKER_IMAGE_NAME=$1
    shift
else
    echo "Error: Missing DOCKER_IMAGE_NAME" >&2
    show_usage >&2
fi

if (( $# )); then
    COMMAND="$@"
fi


if [[ "${COMMAND}" = bash ]]; then
    INTERACTIVE=true
    USE_NET_HOST=true
fi

###############################
### End of argument parsing ###
###############################

source "$(dirname $0)/dev_common.sh" || exit 2

DOCKER_FLAGS=( )
DOCKER_ENV=( )
DOCKER_MOUNT=( )
DOCKER_DEVICES=( )


# If the user gave a shortcut defined in the Jenkinsfile, use it.
EXPANDED_SHORTCUT=$(lookup_image_spec "${DOCKER_IMAGE_NAME}")
if [ -n "${EXPANDED_SHORTCUT}" ]; then
    DOCKER_IMAGE_NAME="${EXPANDED_SHORTCUT}"
fi

# Set up working directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE="$(dirname "${SCRIPT_DIR}")"
DOCKER_FLAGS+=( --workdir /workspace )
DOCKER_MOUNT+=( --volume "${WORKSPACE}":/workspace
                --volume "${SCRIPT_DIR}":/docker
              )

# Set up CI-specific environment variables
DOCKER_ENV+=( --env CI_BUILD_HOME=/workspace
              --env CI_BUILD_USER="$(id -u -n)"
              --env CI_BUILD_UID="$(id -u)"
              --env CI_BUILD_GROUP="$(id -g -n)"
              --env CI_BUILD_GID="$(id -g)"
              --env CI_PYTEST_ADD_OPTIONS="${CI_PYTEST_ADD_OPTIONS:-}"
              --env CI_IMAGE_NAME="${DOCKER_IMAGE_NAME}"
            )


# Pass tvm test data folder through to the docker container, to avoid
# repeated downloads.
TEST_DATA_PATH="${TVM_DATA_ROOT_PATH:-${HOME}/.tvm_test_data}"
DOCKER_MOUNT+=( --volume "${TEST_DATA_PATH}":/workspace/.tvm_test_data )


# Remove the container once it finishes running (--rm) and share the
# PID namespace (--pid=host).  The process inside does not have pid 1
# and SIGKILL is propagated to the process inside, allowing jenkins to
# kill it if needed.
DOCKER_FLAGS+=( --rm --pid=host)

# Expose services running in container to the host.
if $USE_NET_HOST; then
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        DOCKER_FLAGS+=( "-p 8888:8888" )
    else
        DOCKER_FLAGS+=('--net=host')
    fi
fi

# Set up interactive sessions
if ${INTERACTIVE}; then
    DOCKER_FLAGS+=( --interactive --tty )
fi

# Expose external directories to the docker container
for MOUNT_DIR in "${MOUNT_DIRS[@]}"; do
    DOCKER_MOUNT+=( --volume "${MOUNT_DIR}:${MOUNT_DIR}" )
done

# Use nvidia-docker for GPU container.  If nvidia-docker is not
# available, fall back to using "--gpus all" flag, requires docker
# version 19.03 or higher.
if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* || "${DOCKER_IMAGE_NAME}" == *"cuda"* ]]; then
    if type nvidia-docker 1> /dev/null 2> /dev/null; then
        DOCKER_BINARY=nvidia-docker
    else
        DOCKER_BINARY=docker
        DOCKER_FLAGS+=( --gpus all )
    fi
else
    DOCKER_BINARY=docker
fi

# Pass any restrictions of allowed CUDA devices from the host to the
# docker container.
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    DOCKER_ENV+=( --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" )
fi



# Set TVM import path inside the docker image
if [[ "${DOCKER_IMAGE_NAME}" == *"ci"* ]]; then
    DOCKER_ENV+=( "--env" "PYTHONPATH=/workspace/python" )
fi



# If the Vitis-AI docker image is selected, expose the Xilinx FPGA
# devices and required volumes containing e.g. DSA's and overlays
if [[ "${DOCKER_IMAGE_NAME}" == *"demo_vitis_ai"* && -d "/dev/shm" && -d "/opt/xilinx/dsa" && -d "/opt/xilinx/overlaybins" ]]; then
    DOCKER_MOUNT+=( --volume /dev/shm:/dev/shm
                    --volume /opt/xilinx/dsa:/opt/xilinx/dsa
                    --volume /opt/xilinx/overlaybins:/opt/xilinx/overlaybins
                  )

    XCLMGMT_DRIVER="$(find /dev -name xclmgmt\*)"
    for DRIVER in "${XCLMGMT_DRIVER}"; do
       DOCKER_DEVICES+=( --device="${DRIVER}" )
    done

    RENDER_DRIVER="$(find /dev/dri -name renderD\*)"
    for DRIVER in "${RENDER_DRIVER}"; do
        DOCKER_DEVICES+=( --device="${DRIVER}" )
    done
fi

# Add ROCm devices and set ROCM_ENABLED=1 which is used in the with_the_same_user script
# to add the user to the video group
if [[ "${DOCKER_IMAGE_NAME}" == *"rocm"* && -d "/dev/dri" ]]; then
    DOCKER_DEVICES+=( --device=/dev/kfd --device=/dev/dri )
    DOCKER_ENV+=( --env ROCM_ENABLED=1 )
fi

# When running from a git worktree, also mount the original git dir.
if [ -f "${WORKSPACE}/.git" ]; then
    git_dir=$(cd ${WORKSPACE} && git rev-parse --git-common-dir)
    if [ "${git_dir}" != "${WORKSPACE}/.git" ]; then
        DOCKER_MOUNT+=( --volume "${git_dir}:${git_dir}" )
    fi
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

DOCKER_CMD=(${DOCKER_BINARY} run
            "${DOCKER_FLAGS[@]}"
            "${DOCKER_ENV[@]}"
            "${DOCKER_MOUNT[@]}"
            "${DOCKER_DEVICES[@]}"
            "${DOCKER_IMAGE_NAME}"
            bash --login /docker/with_the_same_user
            "${COMMAND[@]}"
           )

if ${DRY_RUN}; then
    echo "${DOCKER_CMD[@]}"
else
    "${DOCKER_CMD[@]}"
fi
