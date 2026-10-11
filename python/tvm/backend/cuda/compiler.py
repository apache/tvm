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
"""Compile one CUDA source group using explicit backend settings."""

from typing import NamedTuple

from ..config import copy_backend_config
from .backend_config import compiler_options, resolve_backend_config


class CompilationResult(NamedTuple):
    """Device artifact and the effective settings used to produce it."""

    binary: bytes
    target_format: str
    config: dict


def compile_source(source: str, backend_config=None, *, target=None) -> CompilationResult:
    """Compile source with per-backend overrides and optional Target/tag defaults.

    When target is omitted, use the current CUDA Target or detect the device.
    Compiler logs are reported on failure and are not part of the artifact.
    """
    from tvm.support import nvcc

    config = resolve_backend_config(copy_backend_config(backend_config).get("cuda"), target)
    use_nvshmem = "#include <nvshmem.h>" in source or "#include <nvshmemx.h>" in source
    output_format = config.get("target_format") or (
        "cubin" if config["compiler"] == "nvrtc" or use_nvshmem else "fatbin"
    )
    if use_nvshmem and output_format != "cubin":
        raise ValueError("NVSHMEM device linking requires target_format='cubin'")
    config["target_format"] = output_format
    driver = getattr(nvcc, f"_compile_cuda_{config['compiler']}")
    binary, _log = driver(
        source, output_format, config["arch"], compiler_options(config), use_nvshmem=use_nvshmem
    )
    return CompilationResult(bytes(binary), output_format, {"cuda": config})
