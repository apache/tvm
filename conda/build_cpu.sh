#!/bin/sh
set -e
conda build --output-folder=conda/pkg -m conda/conda_build_config.yaml conda/tvm-libs
conda build --output-folder=conda/pkg -m conda/conda_build_config.yaml conda/tvm
conda build --output-folder=conda/pkg -m conda/conda_build_config.yaml conda/topi
conda build --output-folder=conda/pkg -m conda/conda_build_config.yaml conda/nnvm
