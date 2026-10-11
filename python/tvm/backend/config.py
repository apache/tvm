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
"""Copy, merge, and serialize explicit per-backend compilation settings."""

import argparse
import json
from collections.abc import Mapping


def copy_backend_config(config=None):
    """Validate and snapshot a mapping without filling any defaults."""
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("backend_config must be a mapping of backend names to configurations")
    result = {}
    for name, value in config.items():
        if name != "cuda":
            raise ValueError(f"backend_config does not yet support backend {name!r}")
        from .cuda.backend_config import validate_backend_config

        result[name] = validate_backend_config(value)
    return result


def merge_backend_configs(*configs):
    """Overlay explicitly supplied keys; argument lists replace inherited lists."""
    result = {}
    for config in configs:
        for name, value in copy_backend_config(config).items():
            result.setdefault(name, {}).update(value)
    return result


def backend_config_json(config=None):
    """Return deterministic JSON, preserving the order of toolchain arguments."""
    return json.dumps(copy_backend_config(config), sort_keys=True, separators=(",", ":"))


def parse_backend_config(value):
    """Read a configuration snapshot without consulting defaults or process state."""
    return copy_backend_config(json.loads(value) if value else None)


def prepare_target(target, config=None, mod=None):
    """Resolve the build target, including architectures supplied by CUDA entries."""
    import tvm_ffi

    from tvm.target import Target

    from .cuda.backend_config import reject_legacy_compile_environment

    reject_legacy_compile_environment()
    config = copy_backend_config(config)
    active = Target.current() if target is None else target
    if isinstance(active, Target) and active.kind.name == "cuda" and "arch" not in active.attrs:
        active = dict(active.export())
    generic_cuda = (
        active is None
        or active == "cuda"
        or (isinstance(active, Mapping) and active.get("kind") == "cuda" and "arch" not in active)
    )
    entries = []
    if mod is not None and generic_cuda:
        from tvm.ir import RegionStmt

        def visit(node):
            if isinstance(node, RegionStmt) and node.op.name == "tirx.device_entry":
                entries.append(parse_backend_config(node.attrs.get("backend_config", "")))

        tvm_ffi.structural_walk(mod, visit)
    if not config and not any(entries):
        return target
    cuda = config.get("cuda", {})
    if generic_cuda:
        arch = cuda.get("arch")
        if arch is None and entries and all(c.get("cuda", {}).get("arch") for c in entries):
            arch = entries[0]["cuda"]["arch"]
        attrs = dict(active) if isinstance(active, Mapping) else {"kind": "cuda"}
        if arch is not None:
            attrs["arch"] = arch
        return Target(attrs)
    active = Target(active)
    if cuda and active.kind.name != "cuda":
        raise ValueError("CUDA backend_config requires a CUDA target")
    arch = cuda.get("arch")
    if arch is not None and active.attrs.get("arch") != arch:
        raise ValueError(
            f"Target.arch={active.attrs.get('arch')!r} conflicts with backend_config arch={arch!r}"
        )
    return active


def argparse_backend_config(value):
    """Parse the same nested mapping accepted by compile and device_entry."""
    try:
        return parse_backend_config(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_backend_config_argument(parser):
    """Add the shared JSON configuration argument to a command-line interface."""
    parser.add_argument(
        "--backend-config",
        type=argparse_backend_config,
        default=None,
        metavar="JSON",
        help='Backend settings, e.g. \'{"cuda":{"compiler":"nvcc","nvcc":["--ftz=false"]}}\'',
    )
