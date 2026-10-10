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
"""Explicit, immutable CUDA compilation settings shared by every compile entry point.

``None`` means unspecified. Entry settings override build settings field by field;
sequences replace inherited sequences, including when the replacement is empty.
"""

import json
import os
import re
from dataclasses import dataclass, field, fields, replace

import tvm_ffi


def _option(kind, *, flag=None, choices=(), minimum=None, maximum=None, default=None):
    return field(
        default=None,
        metadata=dict(
            kind=kind, flag=flag, choices=choices, minimum=minimum, maximum=maximum, default=default
        ),
    )


@dataclass(frozen=True, kw_only=True)
class CompileConfig:
    """CUDA compile defaults or per-device-entry overrides.

    ``arch`` is a real architecture (for example ``sm_100a``). ``compiler`` is
    ``nvrtc`` or ``nvcc``; their default output formats are cubin and fatbin.
    Fast math defaults to enabled. Explicit ``ftz``, ``prec_div``, ``prec_sqrt``
    and ``fmad`` fields override the fast-math preset. Raw compiler/ptxas options
    may only describe options without a structured field. ``dump_dir`` controls
    diagnostics only and never changes the generated binary or its cache key.
    """

    arch: str | None = _option("arch")
    compiler: str | None = _option("str", choices=("nvrtc", "nvcc"), default="nvrtc")
    target_format: str | None = _option("str", choices=("ptx", "cubin", "fatbin"))
    fast_math: bool | None = _option("bool", default=True)
    ftz: bool | None = _option("bool", flag="--ftz")
    prec_div: bool | None = _option("bool", flag="--prec-div")
    prec_sqrt: bool | None = _option("bool", flag="--prec-sqrt")
    fmad: bool | None = _option("bool", flag="--fmad")
    cxx_standard: str | None = _option(
        "str", flag="--std", choices=("c++11", "c++14", "c++17", "c++20")
    )
    device_debug: bool | None = _option("bool", flag="--device-debug", default=False)
    lineinfo: bool | None = _option("bool", flag="--generate-line-info", default=False)
    ptxas_opt_level: int | None = _option("int", flag="--opt-level", minimum=0, maximum=3)
    ptxas_reg_usage_level: int | None = _option(
        "int", flag="--register-usage-level", minimum=0, maximum=10, default=10
    )
    include_dirs: tuple[str, ...] | None = _option("list", flag="--include-path", default=())
    defines: tuple[str, ...] | None = _option("list", flag="--define-macro", default=())
    nvcc_options: tuple[str, ...] | None = _option("list", default=())
    nvrtc_options: tuple[str, ...] | None = _option("list", default=())
    ptxas_options: tuple[str, ...] | None = _option("list", default=())
    dump_dir: str | None = _option("str")

    def __post_init__(self):
        for spec in fields(self):
            value, meta = getattr(self, spec.name), spec.metadata
            if value is None:
                continue
            kind = meta["kind"]
            if kind == "list":
                if not isinstance(value, tuple | list) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise TypeError(f"CompileConfig.{spec.name} must be a sequence of strings")
                object.__setattr__(self, spec.name, tuple(value))
            elif kind == "bool":
                if type(value) is not bool:
                    raise TypeError(f"CompileConfig.{spec.name} must be a Python bool")
            elif kind == "int":
                if type(value) is not int:
                    raise TypeError(f"CompileConfig.{spec.name} must be a Python int")
                if not meta["minimum"] <= value <= meta["maximum"]:
                    raise ValueError(
                        f"CompileConfig.{spec.name} must be between "
                        f"{meta['minimum']} and {meta['maximum']}"
                    )
            elif not isinstance(value, str):
                raise TypeError(f"CompileConfig.{spec.name} must be a string")
            if meta["choices"] and value not in meta["choices"]:
                raise ValueError(f"CompileConfig.{spec.name} must be one of {meta['choices']}")
            if kind == "arch" and not re.fullmatch(r"sm_[0-9]+[af]?", value):
                raise ValueError(
                    "CompileConfig.arch must be a real CUDA architecture, e.g. sm_100a"
                )
        if self.compiler == "nvrtc" and self.target_format == "fatbin":
            raise ValueError("NVRTC cannot produce fatbin; use compiler='nvcc'")
        self._validate_raw_options()

    def _validate_raw_options(self):
        # Reject aliases as well as the canonical spelling. These options must
        # have exactly one source of truth and cannot override resolved fields.
        owned = {s.metadata["flag"] for s in fields(self) if s.metadata["flag"]}
        owned.update(
            (
                "-arch",
                "--gpu-architecture",
                "-gencode",
                "--generate-code",
                "-code",
                "--gpu-code",
                "-optf",
                "--options-file",
                "--use_fast_math",
                "-use_fast_math",
                "-ftz",
                "-prec-div",
                "-prec-sqrt",
                "-fmad",
                "-std",
                "-G",
                "-lineinfo",
                "-register-usage-level",
                "-gpu-architecture",
                "-include-path",
                "-define-macro",
                "-device-debug",
                "-generate-line-info",
                "-I",
                "-D",
                "-O",
                "-Xptxas",
                "--ptxas-options",
                "--cubin",
                "--ptx",
                "--fatbin",
                "-cubin",
                "-ptx",
                "-fatbin",
                "-o",
                "--output-file",
                "-opt-level",
                "-maxrregcount",
                "--maxrregcount",
            )
        )
        for name in ("nvcc_options", "nvrtc_options", "ptxas_options"):
            for value in getattr(self, name) or ():
                key = value.split("=", 1)[0]
                if key in owned or re.match(r"^-[IDO](?:[^-].*)$", key):
                    raise ValueError(
                        f"{name}: {value!r} duplicates a structured CompileConfig field"
                    )

    def with_overrides(self, **updates):
        """Return a validated copy; None removes an explicit override."""
        return replace(self, **updates)

    def overlay(self, overrides):
        """Apply only the fields explicitly set in another CompileConfig."""
        if not isinstance(overrides, CompileConfig):
            raise TypeError("compile_config must be a CUDA CompileConfig")
        return self.with_overrides(**overrides.to_dict())

    def to_dict(self):
        """Return a JSON-compatible mapping preserving explicit false/empty values."""
        return {
            s.name: list(v) if s.metadata["kind"] == "list" else v
            for s in fields(self)
            if (v := getattr(self, s.name)) is not None
        }

    def to_json(self):
        """Versioned, deterministic representation for IR and artifacts."""
        return json.dumps(
            {"version": 1, "options": self.to_dict()}, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, value):
        """Read a versioned compile configuration without consulting process state."""
        payload = json.loads(value)
        if payload.get("version") != 1:
            raise ValueError("Unsupported CUDA CompileConfig version; regenerate the artifact")
        return cls(**payload["options"])

    def resolved(self, target=None):
        """Fill defaults and architecture from an explicit CUDA Target."""
        config = CompileConfig(
            **{
                s.name: s.metadata["default"]
                for s in fields(self)
                if s.metadata["default"] is not None
            }
        ).overlay(self)
        arch = getattr(target, "arch", None) if target is not None else None
        if config.arch is None:
            if arch is None:
                raise ValueError(
                    "CUDA compilation requires CompileConfig(arch=...) or a CUDA Target"
                )
            config = config.with_overrides(arch=arch)
        return config

    def compiler_options(self):
        """Translate the field specification to frontend and ptxas arguments."""
        config = self.resolved()
        options = []
        if config.fast_math:
            options.append("--use_fast_math")
        for spec in fields(config):
            value, flag = getattr(config, spec.name), spec.metadata["flag"]
            if value is None or flag is None or spec.name.startswith("ptxas_"):
                continue
            if spec.name in ("device_debug", "lineinfo"):
                if value:
                    options.append(flag)
            elif spec.metadata["kind"] == "bool":
                options.append(f"{flag}={str(value).lower()}")
            elif spec.metadata["kind"] == "list":
                options.extend(f"{flag}={v}" for v in value)
            else:
                options.append(f"{flag}={value}")
        ptxas = ["-v", "--warn-on-local-memory-usage"]
        for spec in fields(config):
            if spec.name.startswith("ptxas_") and spec.metadata["flag"]:
                value = getattr(config, spec.name)
                if value is not None:
                    ptxas.append(f"{spec.metadata['flag']}={value}")
        ptxas.extend(config.ptxas_options)
        if config.compiler == "nvcc":
            options.append(f"--ptxas-options={','.join(ptxas)}")
        else:
            options.extend(f"--ptxas-options={v}" for v in ptxas)
        options.extend(getattr(config, f"{config.compiler}_options"))
        return options


def pack_compile_config(config):
    """Validate and serialize a public configuration into static IR metadata."""
    if not isinstance(config, CompileConfig):
        raise TypeError("compile_config must be a CUDA CompileConfig")
    return config.to_json()


def prepare_target(target, config, mod=None):
    """Resolve compile-entry Target/CompileConfig without conflicting arch sources."""
    from tvm.target import Target

    reject_legacy_compile_environment()
    if config is not None:
        pack_compile_config(config)
    active = Target.current() if target is None else target
    generic_cuda = (
        active is None
        or active == "cuda"
        or (isinstance(active, dict) and active.get("kind") == "cuda" and "arch" not in active)
    )
    entry_configs = []
    if mod is not None and generic_cuda:
        from tvm.ir import RegionStmt

        def visit(node):
            if isinstance(node, RegionStmt) and node.op.name == "tirx.device_entry":
                value = node.attrs.get("cuda.compile_config")
                entry_configs.append(CompileConfig.from_json(value) if value else CompileConfig())

        tvm_ffi.structural_walk(mod, visit)
    if config is None and not any(c.to_dict() for c in entry_configs):
        return target
    config = config or CompileConfig()
    if generic_cuda:
        arch = config.arch
        if arch is None and entry_configs and all(c.arch for c in entry_configs):
            # Every device entry is explicit; the function's temporary CUDA target
            # only seeds host/device splitting and never replaces an entry's arch.
            arch = entry_configs[0].arch
        attrs = dict(active) if isinstance(active, dict) else {"kind": "cuda"}
        return Target(dict(attrs, **({"arch": arch} if arch else {})))
    active = Target(active)
    if active.kind.name != "cuda":
        raise ValueError("CUDA CompileConfig requires a CUDA target")
    if config.arch is not None and active.arch != config.arch:
        raise ValueError(
            f"Target.arch={active.arch!r} conflicts with CompileConfig.arch={config.arch!r}"
        )
    return active


@tvm_ffi.register_global_func("cuda.resolve_compile_config")
def _resolve_entry(local, defaults, target):
    from tvm.target import Target

    inherited = CompileConfig.from_json(defaults) if defaults else CompileConfig()
    overrides = CompileConfig.from_json(local) if local else CompileConfig()
    config = inherited.overlay(overrides).resolved(target)
    attrs = dict(target.export())
    attrs["arch"] = config.arch
    return [Target(attrs), config.to_json()]


def reject_legacy_compile_environment():
    """Diagnose removed options rather than silently changing a build's behavior."""
    legacy = {
        "TVM_CUDA_COMPILE_MODE": "compiler",
        "TVM_CUDA_NVCC_NO_FAST_MATH": "fast_math=False",
        "TVM_CUDA_NVRTC_EXTRA_OPTS": "nvrtc_options or a structured field",
        "TVM_CUDA_PTXAS_REG_LEVEL": "ptxas_reg_usage_level",
        "TVM_CUDA_PTXAS_EXTRA_OPTS": "ptxas_options or a structured field",
        "TIRX_PREPARE_CUDA_ARCH": "arch",
        "FA4_REG_LEVEL": "ptxas_reg_usage_level",
        "FA4FP4_REG_LEVEL": "ptxas_reg_usage_level",
    }
    for name, field_name in legacy.items():
        if name in os.environ:
            raise ValueError(f"{name} was removed; pass CompileConfig({field_name}=...) explicitly")


def argparse_compile_config(value):
    """Parse a CLI JSON object using exactly the public CompileConfig fields."""
    import argparse

    try:
        return CompileConfig(**json.loads(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_compile_config_argument(parser):
    """Add the same explicit --compile-config option to compile/run/remote CLIs."""
    parser.add_argument(
        "--compile-config",
        type=argparse_compile_config,
        default=None,
        metavar="JSON",
        help='CUDA CompileConfig fields, e.g. \'{"compiler":"nvcc","ftz":false}\'',
    )
