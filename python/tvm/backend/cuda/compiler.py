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
"""Single CUDA source-to-binary service for JIT, offline builds and cuda_host."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .compile_config import CompileConfig, reject_legacy_compile_environment


@dataclass(frozen=True)
class CompilationResult:
    """A compiled device group, including the exact settings needed to reproduce it."""

    binary: bytes
    target_format: str
    config: CompileConfig
    log: str
    cache_key: str


def _toolchain_version(compiler):
    if compiler == "nvcc":
        return subprocess.check_output(["nvcc", "--version"], text=True)
    from cuda.bindings import nvrtc

    result, major, minor = nvrtc.nvrtcVersion()
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Cannot query NVRTC version: {result}")
    return f"nvrtc {major}.{minor}"


def compile_source(source: str, compile_config: CompileConfig) -> CompilationResult:
    """Compile with no mutable process-wide settings or callback replacement.

    The cache identity includes the effective options, source and toolchain
    version. Diagnostics are excluded. The caller may cache this result using
    ``cache_key``; compilation itself does not share mutable cache state.
    """
    from tvm.support import nvcc

    if not isinstance(compile_config, CompileConfig):
        raise TypeError("compile_config must be a CUDA CompileConfig")
    reject_legacy_compile_environment()
    config = compile_config.resolved()
    use_nvshmem = "#include <nvshmem.h>" in source or "#include <nvshmemx.h>" in source
    output_format = config.target_format or (
        "cubin" if config.compiler == "nvrtc" or use_nvshmem else "fatbin"
    )
    if use_nvshmem and output_format != "cubin":
        raise ValueError("NVSHMEM device linking requires target_format='cubin'")
    config = config.with_overrides(target_format=output_format)
    options = config.compiler_options()
    version = _toolchain_version(config.compiler)
    identity = config.with_overrides(dump_dir=None).to_json() + "\0" + version + "\0" + source
    key = hashlib.sha256(identity.encode()).hexdigest()
    driver = getattr(nvcc, f"_compile_cuda_{config.compiler}")
    binary, log = driver(source, output_format, config.arch, options, use_nvshmem=use_nvshmem)
    result = CompilationResult(bytes(binary), output_format, config, log, key)
    if config.dump_dir is not None:
        directory = Path(config.dump_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.cu").write_text(source)
        (directory / f"{key}.{output_format}").write_bytes(result.binary)
        (directory / f"{key}.json").write_text(config.to_json())
        (directory / f"{key}.log").write_text(log)
    return result
