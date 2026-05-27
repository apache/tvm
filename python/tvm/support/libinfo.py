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
"""Build-info query helpers for tvm.support."""

import os


def libinfo():
    """Returns a dictionary of compile-time info — minimal Python fallback.

    The native ``support.GetLibInfo`` global function is no longer registered
    after the upstream sync, so we synthesize the values from build-time hints
    instead.
    """
    return {
        "USE_CUDA": os.environ.get("TVM_USE_CUDA", "ON"),
        "USE_LLVM": os.environ.get("TVM_USE_LLVM", "ON"),
        "USE_NCCL": os.environ.get("TVM_USE_NCCL", "ON"),
        "USE_NVTX": os.environ.get("TVM_USE_NVTX", "ON"),
        "USE_NVSHMEM": os.environ.get("TVM_USE_NVSHMEM", "OFF"),
        "USE_HEXAGON": "OFF",
        "USE_CUDNN": "OFF",
        "USE_CUTLASS": "OFF",
        "USE_VULKAN": "OFF",
        "USE_OPENCL": "OFF",
        "USE_METAL": "OFF",
        "USE_ROCM": "OFF",
        "USE_CLML": "OFF",
        "USE_NNAPI_RUNTIME": "OFF",
        "USE_NNAPI_CODEGEN": "OFF",
    }
