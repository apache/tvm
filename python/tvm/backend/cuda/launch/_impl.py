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
"""Validation and operand packing for the generated CUDA configuration API."""

import math
from dataclasses import fields, replace
from numbers import Integral, Real

from .table import COMPOSITES, KERNEL_FIELDS, LAUNCH_FIELDS


def _integer(value):
    if isinstance(value, Integral):
        return int(value)
    # Avoid importing TVM while the generated configuration module is loaded.
    from tvm.ir.prim import IntImm

    return int(value) if isinstance(value, IntImm) else None


def _scalar(value, spec, name):
    from tvm import ir

    if spec.kind == "enum" and isinstance(value, str):
        choices = dict(spec.enums)
        if value not in choices:
            raise ValueError(f"{name}: expected one of {tuple(choices)}, got {value!r}")
        return choices[value]
    if spec.kind == "handle":
        if not isinstance(value, ir.Expr) or not isinstance(value.ty, ir.PointerType):
            raise TypeError(f"{name} must be a host pointer or handle expression")
        return value
    if isinstance(value, bool) and spec.kind != "bool":
        raise TypeError(f"{name} expects {spec.kind}, not bool")
    constant = _integer(value)
    if spec.kind == "float" and isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return float(value)
    if constant is not None:
        if spec.kind == "bool" and constant not in (0, 1):
            raise ValueError(f"{name} must be a boolean")
        if spec.kind == "enum" and constant not in dict(spec.enums).values():
            raise ValueError(f"{name}: invalid enum value {constant}")
        return constant
    if not isinstance(value, ir.Expr) or not isinstance(value.ty, ir.PrimType):
        raise TypeError(f"{name} must be a scalar {spec.kind} expression")
    dtype = value.ty.dtype
    if dtype.lanes != 1:
        raise TypeError(f"{name} must be scalar")
    if spec.kind == "float":
        valid = dtype.is_float
    elif spec.kind == "bool":
        valid = dtype.is_bool or dtype.is_integer
    else:
        valid = dtype.is_integer
    if not valid:
        raise TypeError(f"{name} has incompatible type {dtype}")
    return value


def _dimensions(value, name):
    from tvm_ffi import Array

    if not isinstance(value, tuple | list | Array):
        value = (value,)
    else:
        value = tuple(value)
    if not 1 <= len(value) <= 3:
        raise ValueError(f"{name} expects one to three dimensions")
    from .table import Field

    spec = Field(name, "int", "")
    for dimension in value:
        _scalar(dimension, spec, name)
        constant = _integer(dimension)
        if constant is not None and not 0 < constant <= 0xFFFFFFFF:
            raise ValueError(f"{name} dimensions must be positive uint32 values")
    return value


class Config:
    """Shared validation for immutable, generated configuration records."""

    def __post_init__(self):
        if self._config_type == "LaunchConfig":
            specs = LAUNCH_FIELDS
        elif self._config_type == "KernelOptions":
            specs = KERNEL_FIELDS
        else:
            specs = COMPOSITES[self._config_type]
        for spec in specs:
            value = getattr(self, spec.name)
            if value is None:
                if spec.default == "required" or (
                    self._config_type in COMPOSITES and spec.kind == "handle"
                ):
                    raise ValueError(f"{self._config_type}.{spec.name} is required")
                continue
            if spec.kind == "dim3":
                object.__setattr__(self, spec.name, _dimensions(value, spec.name))
            elif spec.kind in COMPOSITES:
                if getattr(value, "_config_type", None) != spec.kind:
                    raise TypeError(f"{spec.name} must be a {spec.kind}")
            else:
                normalized = _scalar(value, spec, spec.name)
                if self._config_type == "KernelOptions":
                    constant = _integer(normalized)
                    if constant is None:
                        raise TypeError(
                            f"KernelOptions.{spec.name} must be a compile-time constant"
                        )
                    if spec.kind == "int" and constant <= 0:
                        raise ValueError(f"KernelOptions.{spec.name} must be positive")
        if self._config_type == "LaunchConfig":
            self._validate_launch()
        elif self._config_type == "KernelOptions":
            if self.max_blocks_per_cluster is not None and self.min_blocks_per_sm is None:
                raise ValueError("max_blocks_per_cluster requires min_blocks_per_sm")
            if self.max_registers_per_thread is not None and (
                self.min_blocks_per_sm is not None
                or self.max_blocks_per_cluster is not None
                or self.required_block_size
            ):
                raise ValueError(
                    "max_registers_per_thread conflicts with launch bounds and required_block_size"
                )
        else:
            self._validate_composite()

    def _validate_composite(self):
        ranges = {
            "MemSyncDomainMap": (("default", 0, 255), ("remote", 0, 255)),
            "AccessPolicyWindow": (("num_bytes", 0, 2**63 - 1),),
            "ProgrammaticEvent": (("flags", 0, 0),),
            "LaunchCompletionEvent": (("flags", 0, 0),),
        }
        for name, lower, upper in ranges[self._config_type]:
            value = _integer(getattr(self, name))
            if value is not None and not lower <= value <= upper:
                raise ValueError(f"{self._config_type}.{name} must be between {lower} and {upper}")
        if self._config_type == "AccessPolicyWindow" and isinstance(self.hit_ratio, Real):
            if not 0 <= self.hit_ratio <= 1:
                raise ValueError("AccessPolicyWindow.hit_ratio must be between 0 and 1")

    def _validate_launch(self):
        static_threads = math.prod(_integer(extent) or 1 for extent in self.block)
        if static_threads > 1024:
            raise ValueError("block must contain at most 1024 threads")
        if self.preferred_cluster is not None and self.cluster is None:
            raise ValueError("preferred_cluster requires an explicit cluster")
        shapes = {
            name: tuple(getattr(self, name) or ())
            for name in ("grid", "cluster", "preferred_cluster")
        }
        for name, shape in list(shapes.items()):
            shapes[name] = shape + (1,) * (3 - len(shape))
        for numerator, denominator in (
            ("grid", "cluster"),
            ("grid", "preferred_cluster"),
            ("preferred_cluster", "cluster"),
        ):
            if getattr(self, numerator) is None or getattr(self, denominator) is None:
                continue
            for a, b in zip(shapes[numerator], shapes[denominator]):
                lhs, rhs = _integer(a), _integer(b)
                if lhs is not None and rhs is not None and lhs % rhs:
                    raise ValueError(
                        f"{numerator} must be divisible by {denominator} in every dimension"
                    )
        if self.preferred_cluster is not None:
            pz, cz = _integer(shapes["preferred_cluster"][2]), _integer(shapes["cluster"][2])
            if pz is not None and cz is not None and pz != cz:
                raise ValueError("preferred_cluster.z must equal cluster.z")
        for name, lower, upper in (
            ("dynamic_smem_bytes", 0, 0xFFFFFFFF),
            ("priority", -(2**31), 2**31 - 1),
            ("preferred_shared_memory_carveout", 0, 100),
        ):
            value = getattr(self, name)
            if value is not None:
                constant = _integer(value)
                if constant is not None and not lower <= constant <= upper:
                    raise ValueError(f"{name} must be between {lower} and {upper}")

    def replace(self, **updates):
        """Return a validated copy with the specified fields replaced."""
        return replace(self, **updates)


def pack_launch(config):
    """Return the static field description and ordinary IR value operands."""
    if getattr(config, "_config_type", None) != "LaunchConfig":
        raise TypeError("launch must be a CUDA LaunchConfig")
    from tvm.script.ir_builder.stmt import _as_expr

    names, values = [], []
    for spec in LAUNCH_FIELDS:
        value = getattr(config, spec.name)
        if value is None:
            continue
        if spec.kind == "dim3":
            for axis, extent in zip("xyz", value + (1,) * (3 - len(value))):
                names.append(f"{spec.name}.{axis}")
                values.append(_as_expr(extent))
        elif spec.kind in COMPOSITES:
            for member in COMPOSITES[spec.kind]:
                names.append(f"{spec.name}.{member.name}")
                values.append(_as_expr(_scalar(getattr(value, member.name), member, names[-1])))
        else:
            names.append(spec.name)
            values.append(_as_expr(_scalar(value, spec, spec.name)))
    return names, values


def pack_options(options, launch=None):
    if options is None:
        return {}
    if getattr(options, "_config_type", None) != "KernelOptions":
        raise TypeError("options must be CUDA KernelOptions")
    if options.min_blocks_per_sm is not None and (
        launch is None or any(_integer(extent) is None for extent in launch.block)
    ):
        raise ValueError("min_blocks_per_sm requires static block dimensions")
    result = {
        field.name: int(getattr(options, field.name))
        for field in fields(options)
        if getattr(options, field.name) is not None
    }
    if options.required_block_size:
        if launch is None:
            raise ValueError("required_block_size needs a LaunchConfig")
        for name in ("block", "cluster"):
            shape = getattr(launch, name) or (1,)
            for axis, dimension in zip("xyz", shape + (1,) * (3 - len(shape))):
                value = _integer(dimension)
                if value is None:
                    raise ValueError(
                        "required_block_size requires static block and cluster dimensions"
                    )
                result[f"required_{name}_{axis}"] = value
        if launch.preferred_cluster is not None:
            cluster = launch.cluster or (1,)
            cluster = cluster + (1,) * (3 - len(cluster))
            preferred = launch.preferred_cluster + (1,) * (3 - len(launch.preferred_cluster))
            if any(_integer(a) != _integer(b) for a, b in zip(cluster, preferred)):
                raise ValueError("required_block_size cannot use a different preferred_cluster")
    return result
