#!/bin/bash

set -e
set -u
set -o pipefail

docker build -t  zhenlohuang/tvm-workspace:latest -f Dockerfile.tvm-workspace .