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


def _native_codegen_registered(global_func_name: str) -> bool:
    """Return whether a native global function used only by one optional build is registered.

    Some CMake flags (e.g. ``USE_CUTLASS``) gate compilation of the C++ file
    that registers a given global function, so the function's presence is a
    direct signal that the corresponding flag was enabled at build time.
    """
    try:
        import tvm_ffi  # pylint: disable=import-outside-toplevel

        return tvm_ffi.get_global_func(global_func_name, allow_missing=True) is not None
    except Exception:  # pylint: disable=broad-except
        return False


def libinfo():
    """Returns a dictionary of compile-time info — minimal Python fallback.

    The native ``support.GetLibInfo`` global function is no longer registered
    after the upstream sync, so we synthesize the values from build-time hints
    instead. Where an optional component's registration code is only
    compiled in under a specific CMake flag, we probe for one of its global
    functions rather than guessing.
    """
    return {
        "USE_CUDA": os.environ.get("TVM_USE_CUDA", "ON"),
        "USE_LLVM": os.environ.get("TVM_USE_LLVM", "ON"),
        "USE_NCCL": os.environ.get("TVM_USE_NCCL", "ON"),
        "USE_NVTX": os.environ.get("TVM_USE_NVTX", "ON"),
        "USE_NVSHMEM": os.environ.get("TVM_USE_NVSHMEM", "OFF"),
        "USE_HEXAGON": "OFF",
        "USE_CUDNN": "OFF",
        # "relax.ext.cutlass" is the same global function tvm.contrib.cutlass
        # .has_cutlass() checks; it is registered by
        # src/relax/backend/contrib/cutlass/codegen.cc, which is only
        # compiled when USE_CUDA AND USE_CUTLASS are both ON (see
        # cmake/modules/contrib/CUTLASS.cmake).
        "USE_CUTLASS": "ON" if _native_codegen_registered("relax.ext.cutlass") else "OFF",
        "USE_VULKAN": "OFF",
        "USE_OPENCL": "OFF",
        "USE_METAL": "OFF",
        "USE_ROCM": "OFF",
        "USE_CLML": "OFF",
        "USE_NNAPI_RUNTIME": "OFF",
        "USE_NNAPI_CODEGEN": "OFF",
    }
