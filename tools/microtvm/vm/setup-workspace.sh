#!/bin/bash -e

set -e

TVM_HOME="$1"

# TVM
# NOTE: TVM is presumed to be mounted already by Vagrantfile.
cd "${TVM_HOME}"

tools/microtvm/vm/rebuild-tvm.sh

# NOTE: until the dependencies make it into a top-level pyproject.toml file in main,
# use this approach.
cp tools/microtvm/vm/pyproject.toml .

poetry lock
poetry install
poetry run pip3 install -r ~/zephyr/zephyr/scripts/requirements.txt
