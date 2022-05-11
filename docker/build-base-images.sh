#!/bin/bash -eux

# Build base images (one per Python architecture) used in building the remaining TVM docker images.
set -eux

IMAGES=( )
while [ "${1+x}" == "x" ]; do
    IMAGES=( "${IMAGES[@]}" "$(dirname $0)/Dockerfile.base_$1" )
    shift
done

if [ "${#IMAGES}" -eq 0 ]; then
    IMAGES=$(ls -1 $(dirname $0)/Dockerfile.base_*)
fi

for docker_file in "${IMAGES[@]}"; do
    git check-ignore "${docker_file}" && continue || /bin/true
    arch=${docker_file#"$(dirname $0)/Dockerfile.base_"}
    echo "Building base image for architecture ${arch}"
    $(dirname $0)/build.sh "base_${arch}" --platform "${arch}"

    # NOTE: working dir inside docker is repo root.
    $(dirname $0)/bash.sh -it "tvm.base_${arch}:latest" python3 docker/freeze_deps.py \
                 --ci-constraints=docker/ci-constraints.txt \
                 --gen-requirements-py=python/gen_requirements.py \
                 --template-pyproject-toml=pyproject.toml \
                 --output-base=docker/build/base_${arch}

done
