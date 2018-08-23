#!/usr/bin/env bash
#
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
    echo "Usage: docker/bash.sh <CONTAINER_NAME> [COMMAND]"
    exit -1
fi

DOCKER_IMAGE_NAME=("$1")

if [ "$#" -eq 1 ]; then
    COMMAND="bash"
    CI_DOCKER_EXTRA_PARAMS=("-it --net=host")
else
    shift 1
    COMMAND=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

# Use nvidia-docker if the container is GPU.
if [[ "${DOCKER_IMAGE_NAME}" == *"gpu"* ]]; then
    DOCKER_BINARY="nvidia-docker"
else
    DOCKER_BINARY="docker"
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
echo ${DOCKER_BINARY}
${DOCKER_BINARY} run --rm --pid=host\
    -v ${WORKSPACE}:/workspace \
    -v ${SCRIPT_DIR}:/docker \
    -w /workspace \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMAGE_NAME}\
    bash --login /docker/with_the_same_user \
    ${COMMAND[@]}
