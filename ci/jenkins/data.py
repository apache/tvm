#!/usr/bin/env python3
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
import sys

files_to_stash = {
    # Executables and build files needed to run c++ tests
    "cpptest": ["build/cpptest", "build/build.ninja", "build/CMakeFiles/rules.ninja"],
    # Folder for hexagon build
    "hexagon_api": [
        "build/hexagon_api_output",
    ],
    # This library is produced with HIDE_PRIVATE_SYMBOLS=ON
    "tvm_allvisible": ["build/libtvm_allvisible.so"],
    # runtime files
    "tvm_runtime": ["build/libtvm_runtime.so", "build/config.cmake"],
    # compiler files
    "tvm_lib": [
        "build/libtvm.so",
        "build/libtvm_runtime.so",
        "build/lib/libtvm_ffi.so",
        "build/config.cmake",
    ],
    # gpu related compiler files
    "tvm_lib_gpu_extra": [
        "build/3rdparty/libflash_attn/src/libflash_attn.so",
        "build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so",
    ],
}


# AWS info
aws_default_region = "us-west-2"
aws_ecr_url = "dkr.ecr." + aws_default_region + ".amazonaws.com"

# Docker Images
docker_images = {
    "ci_arm": {
        "tag": "tlcpack/ci-arm:20251130-061900-c429a2b1",
        "platform": "ARM",
    },
    "ci_cpu": {
        "tag": "tlcpack/ci-cpu:20251130-061900-c429a2b1",
        "platform": "CPU",
    },
    "ci_gpu": {
        "tag": "tlcpack/ci-gpu:20251130-061900-c429a2b1",
        "platform": "GPU",
    },
    "ci_lint": {
        "tag": "tlcpack/ci-lint:20251130-061900-c429a2b1",
        "platform": "CPU",
    },
    "ci_wasm": {
        "tag": "tlcpack/ci-wasm:20251130-061900-c429a2b1",
        "platform": "CPU",
    },
}

data = {
    "images": [{"name": k, "platform": v["platform"]} for k, v in docker_images.items()],
    "aws_default_region": aws_default_region,
    "aws_ecr_url": aws_ecr_url,
    **{k: v["tag"] for k, v in docker_images.items()},
    **files_to_stash,
}

if __name__ == "__main__":
    # This is used in docker/dev_common.sh to look up image tags
    name = sys.argv[1]
    if name in docker_images:
        print(docker_images[name]["tag"])
    else:
        exit(1)
