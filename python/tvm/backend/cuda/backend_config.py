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
"""CUDA toolchain selection and native argument lists."""

import os
import re
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import tvm_ffi


class BackendConfig(TypedDict, total=False):
    """Optional CUDA overrides. Toolchain arguments are passed as individual argv items."""

    arch: str
    compiler: Literal["nvcc", "nvrtc"]
    target_format: Literal["ptx", "cubin", "fatbin"]
    nvcc: list[str]
    nvrtc: list[str]
    ptxas: list[str]


# Only arguments that conflict with TVM's architecture, artifact, or tool routing
# are reserved. General compiler options and their values belong to the compiler.
_RESERVED = {
    "-arch",
    "--gpu-architecture",
    "-gpu-architecture",
    "--gpu-name",
    "-arch-name",
    "-gencode",
    "--generate-code",
    "-code",
    "--gpu-code",
    "-o",
    "--output-file",
    "-optf",
    "--options-file",
    "--cubin",
    "-cubin",
    "--ptx",
    "-ptx",
    "--fatbin",
    "-fatbin",
    "-Xptxas",
    "--ptxas-options",
}


def validate_backend_config(config):
    """Check fixed keys and types, and return independently owned containers."""
    if not isinstance(config, Mapping):
        raise TypeError("CUDA backend_config must be a mapping")
    result = {}
    for name, value in config.items():
        if name not in BackendConfig.__annotations__:
            raise ValueError(f"Unknown CUDA backend_config key: {name!r}")
        if name in ("nvcc", "nvrtc", "ptxas"):
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise TypeError(f"CUDA backend_config.{name} must be a sequence of strings")
            if not all(isinstance(v, str) for v in value):
                raise TypeError(f"CUDA backend_config.{name} must be a sequence of strings")
            for item in value:
                if item.split("=", 1)[0] in _RESERVED:
                    raise ValueError(
                        f"{name}: {item!r} is managed by backend_config or the compiler adapter"
                    )
            result[name] = list(value)
        elif not isinstance(value, str):
            raise TypeError(f"CUDA backend_config.{name} must be a string")
        else:
            result[name] = value
    if "arch" in result and not re.fullmatch(r"sm_[0-9]+[af]?", result["arch"]):
        raise ValueError("CUDA backend_config.arch must name a real architecture, e.g. sm_100a")
    if "compiler" in result and result["compiler"] not in ("nvcc", "nvrtc"):
        raise ValueError("CUDA backend_config.compiler must be 'nvcc' or 'nvrtc'")
    if "target_format" in result and result["target_format"] not in ("ptx", "cubin", "fatbin"):
        raise ValueError("CUDA backend_config.target_format must be 'ptx', 'cubin', or 'fatbin'")
    if result.get("compiler") == "nvrtc" and result.get("target_format") == "fatbin":
        raise ValueError("NVRTC cannot produce fatbin; use compiler='nvcc'")
    return result


def default_backend_config():
    """Return fresh CUDA defaults, preserving the existing compilation policy."""
    return BackendConfig(
        compiler="nvrtc",
        nvcc=["--use_fast_math"],
        nvrtc=["--use_fast_math"],
        ptxas=["-v", "--warn-on-local-memory-usage", "--register-usage-level=10"],
    )


def resolve_backend_config(config=None, target=None):
    """Resolve one CUDA configuration after applying target/tag defaults."""
    from tvm.target import Target

    from ..config import copy_backend_config

    reject_legacy_compile_environment()
    overrides = validate_backend_config(config or {})
    if target is None:
        target = Target.current()
    if target is None:
        target = Target(
            {"kind": "cuda", **({"arch": overrides["arch"]} if "arch" in overrides else {})}
        )
    else:
        target = Target(target)
    if target.kind.name != "cuda":
        raise ValueError("CUDA backend_config requires a CUDA target")
    result = default_backend_config()
    result.update(copy_backend_config(target.attrs.get("backend_config", {})).get("cuda", {}))
    result.update(overrides)
    if "arch" not in result:
        arch = target.attrs.get("arch")
        if arch is None:
            raise ValueError(
                "CUDA compilation requires an explicit backend_config arch or Target arch "
                "when no CUDA device is available"
            )
        result["arch"] = str(arch)
    return validate_backend_config(result)


def compiler_options(config):
    """Forward argv to the selected frontend and its ptxas stage."""
    options = list(config[config["compiler"]])
    if config["compiler"] == "nvcc":
        if config["ptxas"]:
            options.append(f"--ptxas-options={','.join(config['ptxas'])}")
    else:
        options.extend(f"--ptxas-options={value}" for value in config["ptxas"])
    return options


@tvm_ffi.register_global_func("cuda.resolve_backend_config")
def _resolve_entry(local, defaults, target):
    from tvm.target import Target

    from ..config import backend_config_json, merge_backend_configs, parse_backend_config

    merged = merge_backend_configs(parse_backend_config(defaults), parse_backend_config(local))
    config = resolve_backend_config(merged.get("cuda"), target)
    attrs = dict(target.export())
    attrs["arch"] = config["arch"]
    return [Target(attrs), backend_config_json({"cuda": config})]


def reject_legacy_compile_environment():
    """Keep removed process-wide compiler policies from silently changing a build."""
    legacy = {
        "TVM_CUDA_COMPILE_MODE": "compiler",
        "TVM_CUDA_NVCC_NO_FAST_MATH": "nvcc",
        "TVM_CUDA_NVRTC_EXTRA_OPTS": "nvrtc",
        "TVM_CUDA_PTXAS_REG_LEVEL": "ptxas",
        "TVM_CUDA_PTXAS_EXTRA_OPTS": "ptxas",
        "TIRX_PREPARE_CUDA_ARCH": "arch",
        "FA4_REG_LEVEL": "ptxas",
        "FA4FP4_REG_LEVEL": "ptxas",
    }
    for name, field in legacy.items():
        if name in os.environ:
            raise ValueError(
                f"{name} was removed; set backend_config['cuda']['{field}'] explicitly"
            )
