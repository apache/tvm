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
# Build a docker container and optionally execute command within a it
#
# Usage: build.sh <CONTAINER_TYPE> [--tag <DOCKER_IMAGE_TAG>]
#                [--dockerfile <DOCKERFILE_PATH>] [-it]
#                [--env <ENVIRONMENT_VARIABLE>]
#                [--net=host] [--cache-from <IMAGE_NAME>] [--cache]
#                [--name CONTAINER_NAME] [--context-path <CONTEXT_PATH>]
#                [--spec DOCKER_IMAGE_SPEC]
#                [--platform <BUILD_PLATFORM>]
#                [<COMMAND>]
#
# CONTAINER_TYPE: Type of the docker container used the run the build,
#                 e.g. "ci_cpu", "ci_gpu"
#
# BUILD_PLATFORM: (Optional) Type of build platform used for the build,
#                 e.g. "arm", "cpu", "gpu". Defaults to "cpu".
#
# DOCKER_IMAGE_TAG: (Optional) Docker image tag to be built and used.
#                   Defaults to 'latest', as it is the default Docker tag.
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.  If
#                  this optional value is not supplied (via the --dockerfile
#                  flag), will use Dockerfile.CONTAINER_TYPE in default
#
# DOCKER_IMAGE_SPEC: Override the default logic to determine the image name and
#                    tag
#
# ENVIRONMENT_VARIABLE: Pass any environment variables through to the container.
#
# IMAGE_NAME: An image to be as a source for cached layers when building the
#             Docker image requested.
#
# CONTAINER_NAME: The name of the docker container, and the hostname that will
#                 appear inside the container.
#
# CONTEXT_PATH: Path to be used for relative path resolution when building
#               the Docker images.
#
# COMMAND (optional): Command to be executed in the docker container
#

DOCKER_ENV=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"

if [[ "$1" == "--tag" ]]; then
    DOCKER_IMAGE_TAG="$2"
    echo "Using custom Docker tag: ${DOCKER_IMAGE_TAG}"
    shift 2
fi

if [[ "$1" == "--dockerfile" ]]; then
    DOCKERFILE_PATH="$2"
    echo "Using custom Dockerfile path: ${DOCKERFILE_PATH}"
    shift 2
fi

if [[ "$1" == "--env" ]]; then
    DOCKER_ENV+=( --env "$2" )
    echo "Setting environment variable: $2"
    shift 2
fi

if [[ "$1" == "-it" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=('-it')
    shift 1
fi

if [[ "$1" == "--spec" ]]; then
    OVERRIDE_IMAGE_SPEC="$2"
    shift 2
fi

if [[ "$1" == "--net=host" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=('--net=host')
    CI_DOCKER_BUILD_EXTRA_PARAMS+=("--network=host")
    shift 1
fi

DOCKER_NO_CACHE_ARG=--no-cache

if [[ "$1" == "--cache-from" ]]; then
    shift 1
    cached_image="$1"
    DOCKER_NO_CACHE_ARG=
    CI_DOCKER_BUILD_EXTRA_PARAMS+=("--cache-from tvm.$CONTAINER_TYPE:$DOCKER_IMAGE_TAG")
    CI_DOCKER_BUILD_EXTRA_PARAMS+=("--cache-from $cached_image")
    shift 1
fi

if [[ "$1" == "--cache" ]]; then
    shift 1
    DOCKER_NO_CACHE_ARG=
fi

if [[ "$1" == "--context-path" ]]; then
    DOCKER_CONTEXT_PATH="$2"
    echo "Using custom context path: ${DOCKER_CONTEXT_PATH}"
    shift 2
else
    DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
    echo "Using default context path: ${DOCKER_CONTEXT_PATH}"
fi

if [[ "$1" == "--name" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=("--name ${2} --hostname ${2}")
    echo "Using container name ${2}"
    shift 2
fi

PLATFORM="cpu"

if [[ "$1" == "--platform" ]]; then
    PLATFORM="$2"
    echo "Using build platform: ${PLATFORM}"
    shift 2
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

COMMAND=("$@")

# Validate command line arguments.
if [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
    supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
        sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
      echo "Usage: $(basename $0) CONTAINER_TYPE COMMAND"
      echo "       CONTAINER_TYPE can be one of [${supported_container_types}]"
      echo "       COMMAND (optional) is a command (with arguments) to run inside"
      echo "               the container."
      exit 1
fi

# Use nvidia-docker if the container is GPU.
if [[ "${CONTAINER_TYPE}" == *"gpu"* || "${CONTAINER_TYPE}" == *"cuda"* ]]; then
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

# Helper function to traverse directories up until given file is found.
function upsearch () {
    test / == "$PWD" && return || \
        test -e "$1" && echo "$PWD" && return || \
        cd .. && upsearch "$1"
}

# Set up WORKSPACE and BUILD_TAG. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}/../}"
BUILD_TAG="${BUILD_TAG:-tvm}"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"

# Determine the docker image name
DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Compose the full image spec with "name:tag" e.g. "tvm.ci_cpu:v0.03"
DOCKER_IMG_SPEC="${DOCKER_IMG_NAME}:${DOCKER_IMAGE_TAG}"

if [[ -n ${OVERRIDE_IMAGE_SPEC+x} ]]; then
    DOCKER_IMG_SPEC="$OVERRIDE_IMAGE_SPEC"
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[@]}"
echo "COMMAND: ${COMMAND[@]}"
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMG_NAME}"
echo "DOCKER_IMAGE_TAG: ${DOCKER_IMAGE_TAG}"
echo "DOCKER_IMG_SPEC: ${DOCKER_IMG_SPEC}"
echo ""


# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build -t ${DOCKER_IMG_SPEC} \
    ${DOCKER_NO_CACHE_ARG} \
    -f "${DOCKERFILE_PATH}" \
    ${CI_DOCKER_BUILD_EXTRA_PARAMS[@]} \
    "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
    echo "ERROR: docker build failed."
    exit 1
fi

if [[ -n ${COMMAND} ]]; then

    # Run the command inside the container.
    echo "Running '${COMMAND[@]}' inside ${DOCKER_IMG_SPEC}..."

    # By default we cleanup - remove the container once it finish running (--rm)
    # and share the PID namespace (--pid=host) so the process inside does not have
    # pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
    echo ${DOCKER_BINARY}
    ${DOCKER_BINARY} run --rm --pid=host \
        -v ${WORKSPACE}:/workspace \
        ${SSH_AUTH_SOCK:+-v $SSH_AUTH_SOCK:/ssh-agent} \
        -w /workspace \
        ${SSH_AUTH_SOCK:+-e "SSH_AUTH_SOCK=/ssh-agent"} \
        -e "CI_BUILD_HOME=/workspace" \
        -e "CI_BUILD_USER=$(id -u -n)" \
        -e "CI_BUILD_UID=$(id -u)" \
        -e "CI_BUILD_GROUP=$(id -g -n)" \
        -e "CI_BUILD_GID=$(id -g)" \
        -e "CI_PYTEST_ADD_OPTIONS=$CI_PYTEST_ADD_OPTIONS" \
        -e "CI_IMAGE_NAME=${DOCKER_IMAGE_NAME}" \
        -e "PLATFORM=${PLATFORM}" \
        ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"} \
        ${CUDA_ENV}\
        ${CI_DOCKER_EXTRA_PARAMS[@]} \
        ${DOCKER_IMG_SPEC} \
        bash --login docker/with_the_same_user \
        ${COMMAND[@]}

fi
