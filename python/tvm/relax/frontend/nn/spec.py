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
"""Compilation specifications, for example, dynamic shape inputs."""

from __future__ import annotations

import inspect
import typing

if typing.TYPE_CHECKING:
    from .core import Module as nn_module_class

ArgSpecType = typing.Union["Int", "Tensor"]
MethodSpecType = typing.Union["MethodSpec", dict[str, ArgSpecType]]
ModuleSpecType = typing.Union["ModuleSpec", dict[str, MethodSpecType]]
SpecAny = typing.Union["Object", "Int", "Tensor", "Tuple"]


class Int:
    """An integer input"""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "int"


class Tensor:
    """A tensor input with static ndim and dtype, but can have symbolic shapes."""

    shape: list[int | str]
    dtype: str

    def __init__(self, shape: typing.Sequence[int | str], dtype: str) -> None:
        self.shape = list(shape)
        self.dtype = dtype

    def __repr__(self) -> str:
        shape = ", ".join(str(i) for i in self.shape)
        return f"Tensor([{shape}], '{self.dtype}')"


class Object:
    """An non-tensor opaque frontend object."""

    object_type: type

    def __init__(self, object_type: type) -> None:
        self.object_type = object_type

    def __repr__(self) -> str:
        return "object"


class Tuple:
    """A tuple input or a list input"""

    name: str
    elements: list[SpecAny] | tuple[SpecAny, ...]

    def __init__(
        self,
        name: str,
        elements: list[SpecAny] | tuple[SpecAny, ...],
    ) -> None:
        assert isinstance(elements, (tuple, list)), f"Unsupported container type: {type(elements)}"
        self.name = name
        self.elements = elements

    def __repr__(self) -> str:
        return self.elements.__repr__()


class MethodSpec:
    """A spec for a compiled method"""

    method: typing.Callable
    arg_names: list[str]
    arg_specs: list[ArgSpecType]
    param_mode: str  # "plain", "packed", "none"
    effect_mode: str  # "plain", "packed", "none"

    def __init__(
        self,
        method: typing.Callable,
        arg_names: list[str],
        arg_specs: list[ArgSpecType],
        param_mode: str,
        effect_mode: str,
    ):
        if param_mode not in ["plain", "packed", "none"]:
            raise ValueError(f"Invalid param_mode: {param_mode}")
        if effect_mode not in ["plain", "packed", "none"]:
            raise ValueError(f"Invalid effect_mode: {effect_mode}")
        self.method = method
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.param_mode = param_mode
        self.effect_mode = effect_mode

    def _repr(self, name: str) -> str:
        args = ", ".join(
            f"{name}: {spec}"
            for name, spec in zip(
                self.arg_names,
                self.arg_specs,
            )
        )
        return f"{name}({args})"

    def __repr__(self) -> str:
        return self._repr(name="MethodSpec")

    @staticmethod
    def from_raw(spec: MethodSpecType, method: typing.Callable) -> MethodSpec:
        """Create MethodSpec from raw python dictionaries.

        Examples
        --------
        .. code-block:: python

            MethodSpec.from_raw(
                spec={
                    "inputs": spec.Tensor([batch_size, "seq_len"], "int32"),
                    "total_seq_len": "int",
                },
                method=module.prefill,
            )
        """
        if isinstance(spec, MethodSpec):
            return spec
        config: dict[str, typing.Any] = spec.pop("$", {})  # type: ignore[assignment]
        param_mode = config.get("param_mode", "plain")
        effect_mode = config.get("effect_mode", "plain")
        method_signature = inspect.signature(method)
        arg_names = list(method_signature.parameters.keys())
        arg_specs = []

        def _convert_arg_spec(arg_spec, arg_name):
            if arg_spec is Int or arg_spec is int:
                return Int()
            if isinstance(arg_spec, str) and arg_spec == "int":
                return Int()
            if isinstance(arg_spec, (Int, Tensor, Object)):
                return arg_spec
            if isinstance(arg_spec, (tuple, list, Tuple)):
                return Tuple(
                    arg_name,
                    elements=type(arg_spec)(
                        [
                            _convert_arg_spec(arg_spec_i, f"{arg_name}_{i}")
                            for i, arg_spec_i in enumerate(arg_spec)
                        ]
                    ),
                )
            raise TypeError(f"Invalid spec for argument {arg_name}: {arg_spec}")

        for arg_name in arg_names:
            if arg_name in spec:
                arg_spec = spec[arg_name]
                arg_spec = _convert_arg_spec(arg_spec, arg_name)
                arg_specs.append(arg_spec)
        return MethodSpec(
            method,
            arg_names,
            arg_specs,
            param_mode=param_mode,
            effect_mode=effect_mode,
        )

    @staticmethod
    def from_torch(args: list[typing.Any], method: typing.Callable) -> MethodSpec:
        """Converts a list of torch tensors to MethodSpec."""
        from .torch import (
            _method_spec_from_torch,
        )

        return _method_spec_from_torch(args, method)


class ModuleSpec:
    """A spec for a compiled nn.Module"""

    module: nn_module_class
    method_names: list[str]
    method_specs: list[MethodSpec]

    def __init__(
        self,
        module: nn_module_class,
        method_names: list[str],
        method_specs: list[MethodSpec],
    ) -> None:
        self.module = module
        self.method_names = method_names
        self.method_specs = method_specs

    @staticmethod
    def from_raw(spec: ModuleSpecType, module: nn_module_class) -> ModuleSpec:
        """Create ModuleSpec from raw python dictionaries.

        Examples
        --------
        .. code-block:: python

            ModuleSpec.from_raw(
                spec={
                    "prefill": {
                        "inputs": spec.Tensor([batch_size, "seq_len"], "int32"),
                        "total_seq_len": int,
                    },
                    "decode": {
                        "inputs": spec.Tensor([batch_size, 1], "int32"),
                        "total_seq_len": int,
                    },
                    "softmax_with_temperature": {
                        "logits": spec.Tensor([1, 1, config.vocab_size], "float32"),
                        "temperature": spec.Tensor([], "float32"),
                    },
                },
                module=module,
            )
        """
        if isinstance(spec, ModuleSpec):
            return spec
        method_names = list(spec.keys())
        method_specs: list[MethodSpec] = []
        for method_name in method_names:
            method_spec = spec[method_name]
            if isinstance(method_spec, MethodSpec):
                pass
            else:
                method_spec = MethodSpec.from_raw(method_spec, getattr(module, method_name))
            method_specs.append(method_spec)
        return ModuleSpec(module, method_names, method_specs)

    def __repr__(self) -> str:
        return "ModuleSpec:\n" + "\n".join(
            "  " + spec._repr(name)
            for name, spec in zip(
                self.method_names,
                self.method_specs,
            )
        )
