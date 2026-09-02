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
"""CUDA target detection and architecture helpers."""

from tvm_ffi import get_global_func


def arch_from_compute_version(compute_version: str) -> str:
    """Return TVM's preferred CUDA architecture for a compute capability.

    TVM uses architecture-specific targets for Hopper and newer devices so
    that CUDA codegen may emit instructions such as WGMMA and tcgen05.  The
    policy lives in the CUDA target canonicalizer and is shared by explicit
    device detection and compiler fallback paths.
    """
    convert = get_global_func("target.cuda_arch_from_compute_version", allow_missing=False)
    return str(convert(compute_version))


def compute_version_from_arch(arch: str) -> str:
    """Convert an ``sm_XX`` target architecture to dotted compute capability."""
    if not arch.startswith("sm_"):
        raise ValueError(f"Expected a CUDA architecture of the form sm_XX, but got {arch!r}")

    version = arch.removeprefix("sm_")
    suffix = ""
    if version and version[-1].isalpha():
        suffix = f".{version[-1]}"
        version = version[:-1]
    if len(version) < 2 or not version.isdigit():
        raise ValueError(f"Expected a CUDA architecture of the form sm_XX, but got {arch!r}")
    return f"{version[:-1]}.{version[-1]}{suffix}"


def detect_target_from_device(dev):
    """Construct a CUDA target containing the device's scheduling resources."""
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": dev.max_registers_per_block,
            "l2_cache_size_bytes": dev.l2_cache_size_bytes,
            "arch": arch_from_compute_version(dev.compute_version),
        }
    )


__all__ = [
    "arch_from_compute_version",
    "compute_version_from_arch",
    "detect_target_from_device",
]
