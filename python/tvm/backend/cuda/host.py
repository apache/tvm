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
"""Export standalone tvm-ffi host C++ with independently compiled CUDA binaries."""

import json

from tvm_ffi import Module


def export_cuda_host(mod: Module) -> str:
    """Return C++ host source with embedded device binaries.

    Build with a CUDA Target whose host is ``cuda_host``. Each device import
    retains its own architecture and CompileConfig. This export preserves those
    binary boundaries; compiling the returned host source does not invoke a
    device compiler. Link the result with tvm-ffi, cudart and the CUDA driver.
    The resulting library needs only those libraries at runtime.

    Source-only fallback imports are compiled using their serialized settings.
    Binary imports work after a save/load roundtrip, without original source.
    """
    if not isinstance(mod, Module):
        raise TypeError("export_cuda_host expects a runtime Module")
    if mod.kind != "c" or "cu" not in mod.get_write_formats():
        raise ValueError("Expected a source module built with host='cuda_host'")
    host_source = mod.inspect_source()
    if not host_source:
        raise ValueError("CUDA-host module has no source")
    binaries = []
    visited, symbols = set(), set()

    def collect(device_mod):
        if device_mod in visited:
            return
        visited.add(device_mod)
        if device_mod.kind != "cuda":
            raise ValueError(f"Expected a CUDA device import, got {device_mod.kind!r}")
        for imported in device_mod.imports:
            collect(imported)
        binary, fmt, names = device_mod["__tvm_cuda_binary"]()
        if fmt == "cuda":
            from .compile_config import CompileConfig
            from .compiler import compile_source

            config = device_mod.inspect_source("cuda.compile_config")
            if not config:
                raise ValueError("CUDA source artifact has no CompileConfig; regenerate it")
            result = compile_source(bytes(binary).decode(), CompileConfig.from_json(config))
            binary, fmt = result.binary, result.target_format
        if fmt not in ("cubin", "fatbin", "ptx"):
            raise ValueError(f"Cannot embed CUDA format {fmt!r}")
        names = tuple(str(name) for name in names)
        duplicate = symbols.intersection(names)
        if duplicate:
            raise ValueError(f"Duplicate CUDA kernel symbols: {sorted(duplicate)}")
        symbols.update(names)
        binaries.append((bytes(binary), names))

    for imported in mod.imports:
        collect(imported)
    source = [
        "#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1",
        "#include <tvm/ffi/extra/cuda/cubin_launcher.h>",
        "#include <cstring>",
    ]
    for index, (binary, _) in enumerate(binaries):
        # PTX must be NUL terminated; an extra trailing byte is harmless for ELF/fatbin.
        values = [str(value) for value in binary] + ["0"]
        lines = [",".join(values[start : start + 32]) for start in range(0, len(values), 32)]
        source.append(
            f"alignas(64) static const unsigned char tvm_cuda_binary_{index}[] = {{\n"
            + ",\n".join(lines)
            + "\n};"
        )
    source.append("static CUkernel tvm_cuda_host_get_kernel(const char* name) {")
    for index, (_, names) in enumerate(binaries):
        match = " || ".join(f"std::strcmp(name, {json.dumps(name)}) == 0" for name in names)
        if match:
            source.append(
                f"  if ({match}) {{\n"
                f"    static tvm::ffi::CubinModule module(tvm_cuda_binary_{index});\n"
                "    return module.GetKernel(name).GetHandle();\n  }"
            )
    source.append('  TVM_FFI_THROW(ValueError) << "Unknown CUDA kernel: " << name;\n}')
    source.append(host_source)
    return "\n\n".join(source)
