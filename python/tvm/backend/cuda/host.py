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
"""CUDA host and device source assembly."""

from tvm_ffi import Module


def bundle_cuda_host(mod: Module) -> str:
    """Return one CUDA C++ translation unit from a built CUDA-host module.

    Build ``mod`` with a CUDA target whose host is ``"cuda_host"``. Its CUDA
    device imports must retain their CUDA C++ source. Device definitions are
    emitted before the host wrappers that launch them. This function only
    reads the modules and returns source; compilation, loading and writing
    files are left to the caller. Sources are concatenated without rewriting
    declarations or deduplicating helpers, so the imports must be compatible
    within one translation unit and provide the types needed by NVCC's host pass.
    Generated host wrappers use tvm-ffi and CUDA libraries. TVM runtime workspace
    allocation and other unsupported runtime services are rejected during codegen.
    Tensor-map encoding additionally requires linking the CUDA driver library.

    Parameters
    ----------
    mod : tvm.runtime.Module
        The built CUDA-host source module with its CUDA device imports.

    Returns
    -------
    source : str
        CUDA C++ source suitable for compilation with NVCC and tvm-ffi headers.

    Raises
    ------
    ValueError
        If the host or its imports are incompatible, or CUDA source is absent
        (for example, after loading a device module saved as a binary).
    """
    if not isinstance(mod, Module):
        raise TypeError("bundle_cuda_host expects a runtime Module")
    if mod.kind != "c" or "cu" not in mod.get_write_formats():
        raise ValueError("Expected a source module built with host='cuda_host'")
    host_source = mod.inspect_source()
    if not host_source:
        raise ValueError("CUDA-host module has no source")

    sources = []
    visited = set()

    def collect(device_mod):
        if device_mod in visited:
            return
        visited.add(device_mod)
        if device_mod.kind != "cuda":
            raise ValueError(f"Expected a CUDA device import, got {device_mod.kind!r}")
        source = device_mod.inspect_source("cuda")
        if not source:
            raise ValueError("CUDA device import has no CUDA C++ source to bundle")
        for imported in device_mod.imports:
            collect(imported)
        sources.append(source)

    for imported in mod.imports:
        collect(imported)
    return "\n\n".join([*sources, host_source])
