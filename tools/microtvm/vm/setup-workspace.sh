#!/bin/bash -e

set -e

TVM_HOME="$1"

# TVM
# NOTE: TVM is presumed to be mounted already by Vagrantfile.
cd "${TVM_HOME}"

tools/microtvm/vm/rebuild-tvm.sh

poetry install
poetry run pip3 install -r ~/zephyr/zephyr/scripts/requirements.txt
