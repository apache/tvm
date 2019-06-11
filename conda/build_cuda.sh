#!/bin/sh
set -e

conda build --output-folder=conda/pkg --variants "{cuda: True, cuda_version: ${CUDA_VERSION%.*}}" conda/tvm-libs
