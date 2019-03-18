#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update && apt-get install -y --no-install-recommends git cmake

# TODO: specific tag?
git clone https://github.com/Maratyszcza/NNPACK NNPACK
(cd NNPACK && git checkout 1e005b0c2)

mkdir -p NNPACK/build
cd NNPACK/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=. -DNNPACK_INFERENCE_ONLY=OFF -DNNPACK_CONVOLUTION_ONLY=OFF -DNNPACK_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make -j4 && make install
cd -
