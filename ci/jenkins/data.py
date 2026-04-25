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
"""Bundle registry for CI artifact stashing.

Single source of truth for the file lists uploaded to / downloaded from S3 by
``ci/scripts/jenkins/s3.py``. This module deliberately carries nothing else —
docker image tags live in ``ci/jenkins/docker-images.ini`` and Jinja-template
metadata (image platforms, AWS endpoints) lives in ``ci/jenkins/generate.py``.

CLI: ``python3 ci/jenkins/data.py <bundle> [<bundle> ...]`` resolves bundle
names to their file paths (one per line; exit 1 on unknown name). Used by
``s3.py`` at Jenkins runtime and by any external caller that needs
data-driven artifact lists.
"""

import sys

files_to_stash = {
    # Executables and build files needed to run c++ tests
    "cpptest": ["build/cpptest", "build/build.ninja", "build/CMakeFiles/rules.ninja"],
    # Folder for hexagon build
    "hexagon_api": [
        "build/hexagon_api_output",
    ],
    # runtime files
    "tvm_runtime": ["build/libtvm_runtime.so", "build/config.cmake"],
    # compiler files (libtvm_allvisible is the HIDE_PRIVATE_SYMBOLS=ON
    # variant cpptest links against; bundled here so every consumer of
    # tvm_lib gets it without having to remember a second bundle name).
    "tvm_lib": [
        "build/libtvm.so",
        "build/libtvm_runtime.so",
        "build/lib/libtvm_ffi.so",
        "build/libtvm_allvisible.so",
        "build/config.cmake",
    ],
    # gpu related compiler files
    "tvm_lib_gpu_extra": [
        "build/3rdparty/libflash_attn/src/libflash_attn.so",
        "build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so",
    ],
}


if __name__ == "__main__":
    paths = []
    for name in sys.argv[1:]:
        if name not in files_to_stash:
            print(f"unknown bundle: {name}", file=sys.stderr)
            sys.exit(1)
        paths.extend(files_to_stash[name])
    for p in paths:
        print(p)
