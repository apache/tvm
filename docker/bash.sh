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
# Start a bash, mount REPO_MOUNT_POINT to be current directory.
#
# Usage: docker/bash.sh [-i|--interactive] [--net=host] [-t|--tty]
#          [--mount MOUNT_DIR] [--repo-mount-point REPO_MOUNT_POINT]
#          [--dry-run] [--name NAME]
#          <DOCKER_IMAGE_NAME> [--] [COMMAND]
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
Usage: docker/bash.sh [-i|--interactive] [--net=host] [-t|--tty]
         [--mount MOUNT_DIR] [--repo-mount-point REPO_MOUNT_POINT]
         [--dry-run] [--name NAME]
         <DOCKER_IMAGE_NAME> [--] [COMMAND]

-h, --help

    Display this help message.

-i, --interactive

    Start the docker session in interactive mode.

-t, --tty

    Start the docker session with a pseudo terminal (tty).

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

--repo-mount-point REPO_MOUNT_POINT

    The directory inside the docker container at which the TVM
    repository should be mounted, and is used as the workspace inside
    the docker container.

    If unspecified, the mount location depends on the environment.  If
    running inside Jenkins, the mount location will be /workspace.
    Otherwise, the mount location of the repository will be the same
    as the external location of the repository, to maintain
    compatibility with git-worktree.

--no-gpu

    Do not use GPU device drivers even if using an CUDA Docker image

--dry-run

    Print the docker command to be run, but do not execute it.

--env

    Pass an environment variable through to the container.

--name

    Set the name of the docker container, and the hostname that will
    appear inside the container.

DOCKER_IMAGE_NAME

    The name of the docker container to be run.  This can be an
    explicit name of a docker image (e.g. "tlcpack/ci-gpu:v0.76") or
    can be a shortcut as defined in the TVM Jenkinsfile
    (e.g. "ci_gpu").

COMMAND

    The command to be run inside the docker container.  If this is set
    to "bash", the --interactive, --tty and --net=host flags are set.
    If no command is specified, defaults to "bash".  If the command
    contains dash-prefixed arguments, the command should be preceded
    by -- to indicate arguments that are not intended for bash.sh.

EOF
}


#################################
### Start of argument parsing ###
#################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

DRY_RUN=false
INTERACTIVE=false
TTY=false
USE_NET_HOST=false
USE_GPU=true
DOCKER_IMAGE_NAME=
COMMAND=bash
MOUNT_DIRS=( )
CONTAINER_NAME=

# TODO(Lunderberg): Remove this if statement and always set to
# "${REPO_DIR}".  The consistent directory for Jenkins is currently
# necessary to allow cmake build commands to run in CI after the build
# steps.
# TODO(https://github.com/apache/tvm/issues/11952):
# Figure out a better way to keep the same path
# between build and testing stages.
if [[ -n "${JENKINS_HOME:-}" ]]; then
    REPO_MOUNT_POINT=/workspace
else
    REPO_MOUNT_POINT="${REPO_DIR}"
fi


function parse_error() {
    echo "$@" >&2
    show_usage >&2
    exit 1
}

# Handle joined flags, such as interpreting -ih as -i -h.  Either rewrites
# the current argument if it is a joined argument, or shifts all arguments
# otherwise.  Should be called as "eval $break_joined_flag" where joined
# flags are possible.  Can't use a function definition, because it needs
# to overwrite the parent scope's behavior.
break_joined_flag='if (( ${#1} == 2 )); then shift; else set -- -"${1#-i}" "${@:2}"; fi'

DOCKER_ENV=( )
DOCKER_FLAGS=( )

while (( $# )); do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;

        -i*|--interactive)
            INTERACTIVE=true
            eval $break_joined_flag
            ;;

        -t*|--tty)
            TTY=true
            eval $break_joined_flag
            ;;

        --net=host)
            USE_NET_HOST=true
            shift
            ;;

        --net)
            DOCKER_FLAGS+=( --net "$2" )
            shift 2
            ;;

        --mount)
            if [[ -n "$2" ]]; then
                MOUNT_DIRS+=("$2")
                shift 2
            else
                parse_error 'ERROR: --mount requires a non-empty argument'
            fi
            ;;

        --mount=?*)
            MOUNT_DIRS+=("${1#*=}")
            shift
            ;;

        --name)
            if [[ -n "$2" ]]; then
                CONTAINER_NAME="$2"
                shift 2
            else
                parse_error 'ERROR: --name requires a non empty argument'
            fi
            ;;

        --env)
            DOCKER_ENV+=( --env "$2" )
            shift 2
            ;;

        --volume)
            DOCKER_FLAGS+=( --volume "$2" )
            shift 2
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;

        --no-gpu)
            USE_GPU=false
            shift
            ;;

        --repo-mount-point)
            if [[ -n "$2" ]]; then
                REPO_MOUNT_POINT="$2"
                shift 2
            else
                parse_error 'ERROR: --repo-mount-point requires a non-empty argument'
            fi
            ;;

        --repo-mount-point=?*)
            REPO_MOUNT_POINT="${1#*=}"
            shift
            ;;

        --)
            shift
            COMMAND=( "$@" )
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
            # First positional argument is the image name, all
            # remaining below to the COMMAND.
            if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
                DOCKER_IMAGE_NAME=$1
                shift
            else
                COMMAND=( "$@" )
                break
            fi
            ;;
    esac
done

if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
    echo "Error: Missing DOCKER_IMAGE_NAME" >&2
    show_usage >&2
fi

if [[ ${COMMAND[@]+"${COMMAND[@]}"} = bash ]]; then
    INTERACTIVE=true
    TTY=true
    USE_NET_HOST=true
fi



###############################
### End of argument parsing ###
###############################

source "$(dirname $0)/dev_common.sh" || exit 2

DOCKER_MOUNT=( )
DOCKER_DEVICES=( )


# If the user gave a shortcut defined in the Jenkinsfile, use it.
EXPANDED_SHORTCUT=$(lookup_image_spec "${DOCKER_IMAGE_NAME}")
if [ -n "${EXPANDED_SHORTCUT}" ]; then
    if [ "${CI+x}" == "x" ]; then
        DOCKER_IMAGE_NAME="${EXPANDED_SHORTCUT}"
    else
        python3 ci/scripts/determine_docker_images.py "$DOCKER_IMAGE_NAME=$EXPANDED_SHORTCUT" 2> /dev/null
        DOCKER_IMAGE_NAME=$(cat ".docker-image-names/$DOCKER_IMAGE_NAME")
        if [[ "$DOCKER_IMAGE_NAME" == *"tlcpackstaging"* ]]; then
            echo "WARNING: resolved docker image to fallback tag in tlcpackstaging" >&2
        fi
    fi
fi

# Set up working directories

DOCKER_FLAGS+=( --workdir "${REPO_MOUNT_POINT}" )
DOCKER_MOUNT+=( --volume "${REPO_DIR}":"${REPO_MOUNT_POINT}"
                --volume "${SCRIPT_DIR}":/docker
              )

# Set up CI-specific environment variables
DOCKER_ENV+=( --env CI_BUILD_HOME="${REPO_MOUNT_POINT}"
              --env CI_BUILD_USER="$(id -u -n)"
              --env CI_BUILD_UID="$(id -u)"
              --env CI_BUILD_GROUP="$(id -g -n)"
              --env CI_BUILD_GID="$(id -g)"
              --env CI_PYTEST_ADD_OPTIONS="${CI_PYTEST_ADD_OPTIONS:-}"
              --env CI_IMAGE_NAME="${DOCKER_IMAGE_NAME}"
            )

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
        DOCKER_FLAGS+=( -p 8888:8888 )
    else
        DOCKER_FLAGS+=(--net=host)
    fi
fi

# Set up interactive sessions
if ${INTERACTIVE}; then
    DOCKER_FLAGS+=( --interactive )
fi

if ${TTY}; then
    DOCKER_FLAGS+=( --tty )
fi

# Setup the docker name and the hostname inside the container
if [[ ! -z "${CONTAINER_NAME}" ]]; then
    DOCKER_FLAGS+=( --name ${CONTAINER_NAME} --hostname ${CONTAINER_NAME})
fi

# Expose external directories to the docker container
for MOUNT_DIR in ${MOUNT_DIRS[@]+"${MOUNT_DIRS[@]}"}; do
    DOCKER_MOUNT+=( --volume "${MOUNT_DIR}:${MOUNT_DIR}" )
done

# Use nvidia-docker for GPU container.  If nvidia-docker is not
# available, fall back to using "--gpus all" flag, requires docker
# version 19.03 or higher.
if [[ "$USE_GPU" == "true" ]] && [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* || "${DOCKER_IMAGE_NAME}" == *"cuda"* ]]; then
    if type nvidia-docker 1> /dev/null 2> /dev/null; then
        DOCKER_BINARY=nvidia-docker
    else
        DOCKER_BINARY=docker
        DOCKER_FLAGS+=( --gpus all )
    fi

    # nvidia-docker treats Vulkan as a graphics API, so we need to
    # request passthrough of graphics APIs.  This could also be set in
    # the Dockerfile.
    DOCKER_ENV+=( --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility )

    # But as of nvidia-docker version 2.6.0-1, we still need to pass
    # through the nvidia icd files ourselves.
    ICD_SEARCH_LOCATIONS=(
        # https://github.com/KhronosGroup/Vulkan-Loader/blob/master/loader/LoaderAndLayerInterface.md#icd-discovery-on-linux
        /usr/local/etc/vulkan/icd.d
        /usr/local/share/vulkan/icd.d
        /etc/vulkan/icd.d
        /usr/share/vulkan/icd.d
        # https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md#icd-installation
        /etc/glvnd/egl_vendor.d
        /usr/share/glvnd/egl_vendor.d
    )
    for filename in $(find "${ICD_SEARCH_LOCATIONS[@]}" -name "*nvidia*.json" 2> /dev/null); do
        DOCKER_MOUNT+=( --volume "${filename}":"${filename}":ro )
    done

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
    DOCKER_ENV+=( --env PYTHONPATH="${REPO_MOUNT_POINT}"/python )
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
if [ -f "${REPO_DIR}/.git" ]; then
    git_dir=$(cd ${REPO_DIR} && git rev-parse --git-common-dir)
    if [ "${git_dir}" != "${REPO_DIR}/.git" ]; then
        DOCKER_MOUNT+=( --volume "${git_dir}:${git_dir}" )
    fi
fi

# Print arguments.
echo "REPO_DIR: ${REPO_DIR}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo Running \'${COMMAND[@]+"${COMMAND[@]}"}\' inside ${DOCKER_IMAGE_NAME}...

DOCKER_CMD=(${DOCKER_BINARY} run
            ${DOCKER_FLAGS[@]+"${DOCKER_FLAGS[@]}"}
            ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"}
            ${DOCKER_MOUNT[@]+"${DOCKER_MOUNT[@]}"}
            ${DOCKER_DEVICES[@]+"${DOCKER_DEVICES[@]}"}
            "${DOCKER_IMAGE_NAME}"
            bash --login /docker/with_the_same_user
            ${COMMAND[@]+"${COMMAND[@]}"}
           )

if ${DRY_RUN}; then
    echo ${DOCKER_CMD[@]+"${DOCKER_CMD[@]}"}
else
    ${DOCKER_CMD[@]+"${DOCKER_CMD[@]}"}
fi
