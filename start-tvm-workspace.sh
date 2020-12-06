#!/bin/bash

set -e
set -u
set -o pipefail

docker run -d -p 8888:8888 -p 2222:22 -v $PWD:/workspace/tvm --name tvm-workspace zhenlohuang/tvm-workspace:latest
docker exec -it tvm-workspace bash