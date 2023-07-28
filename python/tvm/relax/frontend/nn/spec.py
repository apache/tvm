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
import inspect
import threading
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from tvm import tir
from tvm.ir import IRModule

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo
from . import core

ArgSpecType = Union["Int", "Tensor"]
MethodSpecType = Union["MethodSpec", Dict[str, ArgSpecType]]
ModuleSpecType = Union["ModuleSpec", Dict[str, MethodSpecType]]


class Int:  # pylint: disable=too-few-public-methods
    """An integer input"""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "int"


class Tensor:  # pylint: disable=too-few-public-methods
    """A tensor input with static ndim and dtype, but can have symbolic shapes."""

    shape: List[Union[int, str]]
    dtype: str

    def __init__(self, shape: Sequence[Union[int, str]], dtype: str) -> None:
        self.shape = list(shape)
        self.dtype = dtype

    def __repr__(self) -> str:
        shape = ", ".join(str(i) for i in self.shape)
        return f"Tensor([{shape}], '{self.dtype}')"


class MethodSpec:
    """A spec for a compiled method"""

    method: Callable
    arg_names: List[str]
    arg_specs: List[ArgSpecType]

    def __init__(self, method: Callable, arg_names: List[str], arg_specs: List[ArgSpecType]):
        self.method = method
        self.arg_names = arg_names
        self.arg_specs = arg_specs

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
    def from_raw(spec: MethodSpecType, method: Callable) -> "MethodSpec":
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
        method_signature = inspect.signature(method)
        arg_names = list(method_signature.parameters.keys())
        arg_specs = []
        for arg_name in arg_names:
            arg_spec = spec[arg_name]
            if arg_spec is Int or arg_spec is int:
                arg_spec = Int()
            elif isinstance(arg_spec, str) and arg_spec == "int":
                arg_spec = Int()
            elif isinstance(arg_spec, (Int, Tensor)):
                pass
            else:
                raise TypeError(f"Invalid spec for argument {arg_name}: {arg_spec}")
            arg_specs.append(arg_spec)
        return MethodSpec(method, arg_names, arg_specs)

    @staticmethod
    def from_torch(args: List[Any], method: Callable) -> "MethodSpec":
        """Converts a list of torch tensors to MethodSpec."""
        from .torch import (  # pylint: disable=import-outside-toplevel
            _method_spec_from_torch,
        )

        return _method_spec_from_torch(args, method)

    def as_inputs(self) -> List[Union[tir.Var, core.Tensor]]:
        """Convert the MethodSpec to a list of inputs to Module's method."""
        str2var: Dict[str, tir.Var] = {}

        def _get_var(name: str) -> tir.Var:
            if name in str2var:
                return str2var[name]
            var = tir.Var(name, "int64")
            str2var[name] = var
            return var

        args = []
        for arg_name, arg_spec in zip(self.arg_names, self.arg_specs):
            if isinstance(arg_spec, Int):
                arg = _get_var(arg_name)
            elif isinstance(arg_spec, Tensor):
                arg = core._tensor_placeholder(  # pylint: disable=protected-access
                    name=arg_name,
                    shape=[_get_var(x) if isinstance(x, str) else x for x in arg_spec.shape],
                    dtype=arg_spec.dtype,
                )
            else:
                raise TypeError(f"Invalid spec for argument {arg_name}: {arg_spec}")
            args.append(arg)
        return args


class ModuleSpec:
    """A spec for a compiled nn.Module"""

    module: core.Module
    method_names: List[str]
    method_specs: List[MethodSpecType]

    def __init__(
        self,
        module: core.Module,
        method_names: List[str],
        method_specs: List[MethodSpecType],
    ) -> None:
        self.module = module
        self.method_names = method_names
        self.method_specs = method_specs

    @staticmethod
    def from_raw(spec: ModuleSpecType, module: core.Module) -> "ModuleSpec":
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
        method_specs = []
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
            "  " + spec._repr(name)  # pylint: disable=protected-access
            for name, spec in zip(
                self.method_names,
                self.method_specs,
            )
        )


class SpecBuilder:
    """Builder of ModuleSpec, which exports an nn.Module to TVM IRModule."""

    _tls = threading.local()

    builder: BlockBuilder
    io_effect: core.Effect

    def __init__(self) -> None:
        from .modules import IOEffect  # pylint: disable=import-outside-toplevel

        self.builder = BlockBuilder()
        self.io_effect = IOEffect()

    @staticmethod
    def current() -> "SpecBuilder":
        """Get the current SpecBuilder under the with scope."""
        assert hasattr(SpecBuilder._tls, "current")
        return SpecBuilder._tls.current

    def __enter__(self) -> "SpecBuilder":
        assert not hasattr(SpecBuilder._tls, "current")
        SpecBuilder._tls.current = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert hasattr(SpecBuilder._tls, "current")
        delattr(SpecBuilder._tls, "current")

    def build(self, spec: ModuleSpec) -> Tuple[IRModule, List[Tuple[str, core.Parameter]]]:
        """Build the ModuleSpec to TVM IRModule. Returns the IRModule and the parameters."""

        # pylint: disable=protected-access
        def _params() -> List[Tuple[str, core.Parameter]]:
            params = []
            for name, param in core._attribute_finder(
                spec.module, prefix="", condition_yield=lambda x: isinstance(x, core.Parameter)
            ):
                params.append((name, param))
            return params

        def _effects() -> List[Tuple[str, core.Effect]]:
            result = [("", self.io_effect)]
            for name, effect in core._attribute_finder(
                spec.module, "", condition_yield=lambda x: isinstance(x, core.Effect)
            ):
                result.append((name, effect))
            return result

        # pylint: enable=protected-access

        params = _params()
        effects = _effects()
        with self:
            with self.builder.function("_initialize_effect"):
                with self.builder.dataflow():
                    outputs = _emit_effect_init(self.builder, effects)
                self.builder.emit_func_output(outputs, params=[])
            for method_name, method_spec in zip(spec.method_names, spec.method_specs):
                with self.builder.function(method_name):
                    with self.builder.dataflow():
                        outputs, inputs = _emit_method(self.builder, method_spec, params, effects)
                    self.builder.emit_func_output(outputs, inputs)
        return self.builder.get(), params


def _emit_effect_init(
    builder: BlockBuilder,
    effects: List[Tuple[str, core.Effect]],
):
    outputs = []
    for prefix, effect in effects:
        inits = effect.emit_init(prefix, builder)
        assert isinstance(inits, list)
        outputs.extend(inits)
    outputs = builder.emit_output(builder.emit(rx.Tuple(outputs)))
    return outputs


def _emit_method(
    builder: BlockBuilder,
    spec: MethodSpec,
    params: List[Tuple[str, core.Parameter]],
    effects: List[Tuple[str, core.Effect]],
):
    # pylint: disable=protected-access
    def _unwrap_ret(expr: Any) -> Any:
        if isinstance(expr, core.Tensor):
            return expr._expr  # pylint: disable=protected-access
        if isinstance(expr, tuple):
            return rx.Tuple([_unwrap_ret(x) for x in expr])
        if isinstance(expr, list):
            return rx.Tuple([_unwrap_ret(x) for x in expr])
        raise TypeError(f"Unsupported return type: {type(expr)}")

    def _convert_input(arg):
        if isinstance(arg, tir.Var):
            return rx.Var(arg.name, struct_info=ShapeStructInfo(values=[arg]))
        if isinstance(arg, core.Tensor):
            return arg._expr  # pylint: disable=protected-access
        raise TypeError(f"Unsupported input type: {type(arg)}")

    explicit_inputs = spec.as_inputs()
    inputs = []
    for arg in explicit_inputs:
        inputs.append(_convert_input(arg))
    for name, param in params:
        param._expr = core._tensor_placeholder(name, param.shape, param.dtype)._expr
        inputs.append(param._expr)
    for name, effect in effects:
        inputs.extend(effect.create(name))
    # pylint: enable=protected-access

    outputs = spec.method(*explicit_inputs)
    effect_outputs = []
    for _, effect in effects:
        effect_outputs.extend(effect.finalize())
    outputs = builder.emit_output(rx.Tuple([_unwrap_ret(outputs), rx.Tuple(effect_outputs)]))
    return outputs, inputs
