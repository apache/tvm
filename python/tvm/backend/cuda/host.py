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

import json

from tvm_ffi import Module


def bundle_cuda_host_device(mod: Module) -> str:
    """Return one CUDA C++ translation unit from a built CUDA-host module.

    Build ``mod`` with a CUDA target whose host is ``"cuda_host"``. Its CUDA
    device imports must retain their CUDA C++ source. Device definitions are
    emitted before the host wrappers that launch them. This function only
    reads the modules and returns source; compilation, loading and writing
    files are left to the caller. Multiple imports must retain unmodified
    codegen components, so shared headers and helpers can be emitted once.

    Parameters
    ----------
    mod : tvm.runtime.Module
        The built CUDA-host source module with its CUDA device imports.

    Returns
    -------
    source : str
        CUDA C++ source suitable for compilation with NVCC and TVM headers.

    Raises
    ------
    ValueError
        If the host or its imports are incompatible, or CUDA source is absent
        (for example, after loading a device module saved as a binary).
    """
    if not isinstance(mod, Module):
        raise TypeError("bundle_cuda_host_device expects a runtime Module")
    if mod.kind != "c" or "cu" not in mod.get_write_formats():
        raise ValueError("Expected a source module built with host='cuda_host'")
    host_source = mod.inspect_source()
    if not host_source:
        raise ValueError("CUDA-host module has no source")

    modules = []
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
        modules.append((device_mod, source))

    for imported in mod.imports:
        collect(imported)
    if len(modules) <= 1:
        return "\n\n".join([source for _, source in modules] + [host_source])

    # Combine codegen components instead of parsing CUDA text. A union of the
    # header tags emits shared types/helpers once, including mixed dtype modules.
    from .codegen.header import header_generator  # pylint: disable=import-outside-toplevel

    tags = set()
    symbols = set()
    utilities = {}
    bodies = []
    for device_mod, source in modules:
        metadata = device_mod.inspect_source("cuda.bundle")
        if not metadata:
            raise ValueError(
                "Multiple CUDA imports require unmodified codegen source components; "
                "a device import has missing or postprocessed source metadata"
            )
        try:
            parts = json.loads(metadata)
            original = (
                parts["header"] + "".join(u["code"] for u in parts["utilities"]) + parts["body"]
            )
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError("Invalid CUDA codegen source components") from err
        if source != original:
            raise ValueError("CUDA source no longer matches its codegen components")
        for name in parts["symbols"]:
            if name in symbols or name in utilities:
                raise ValueError(f"Conflicting CUDA symbol in imports: {name}")
            symbols.add(name)
        for utility in parts["utilities"]:
            name, code = utility["name"], utility["code"]
            if name in symbols or (name in utilities and utilities[name] != code):
                raise ValueError(f"Conflicting CUDA helper in imports: {name}")
            utilities[name] = code
        tags.update(parts["header_tags"])
        bodies.append(parts["body"])
    return "\n\n".join([header_generator(sorted(tags)), *utilities.values(), *bodies, host_source])
