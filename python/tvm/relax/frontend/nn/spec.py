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
import typing

from tvm import tir
from tvm.ir import IRModule
from tvm.runtime import Module, load_static_library

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo, TupleStructInfo
from . import core

ArgSpecType = typing.Union["Int", "Tensor"]
MethodSpecType = typing.Union["MethodSpec", typing.Dict[str, ArgSpecType]]
ModuleSpecType = typing.Union["ModuleSpec", typing.Dict[str, MethodSpecType]]
SpecAny = typing.Union["Int", "Tensor", "Tuple"]


class Int:  # pylint: disable=too-few-public-methods
    """An integer input"""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "int"


class Tensor:  # pylint: disable=too-few-public-methods
    """A tensor input with static ndim and dtype, but can have symbolic shapes."""

    shape: typing.List[typing.Union[int, str]]
    dtype: str

    def __init__(self, shape: typing.Sequence[typing.Union[int, str]], dtype: str) -> None:
        self.shape = list(shape)
        self.dtype = dtype

    def __repr__(self) -> str:
        shape = ", ".join(str(i) for i in self.shape)
        return f"Tensor([{shape}], '{self.dtype}')"


class Tuple:  # pylint: disable=too-few-public-methods
    """A tuple input or a list input"""

    name: str
    elements: typing.Union[typing.List[SpecAny], typing.Tuple[SpecAny, ...]]

    def __init__(
        self,
        name: str,
        elements: typing.Union[typing.List[SpecAny], typing.Tuple[SpecAny, ...]],
    ) -> None:
        assert isinstance(elements, (tuple, list)), f"Unsupported container type: {type(elements)}"
        self.name = name
        self.elements = elements

    def __repr__(self) -> str:
        return self.elements.__repr__()


class ConstInt:  # pylint: disable=too-few-public-methods
    """An integer constant"""

    dtype: typing.Optional[str]

    def __init__(self, dtype: str = None) -> None:
        self.dtype = dtype

    def __repr__(self) -> str:
        if self.dtype is None:
            return "const.int"
        return f"const.int({self.dtype})"


class ConstFloat:  # pylint: disable=too-few-public-methods
    """A float constant"""

    dtype: typing.Optional[str]

    def __init__(self, dtype: str = None) -> None:
        self.dtype = dtype

    def __repr__(self) -> str:
        if self.dtype is None:
            return "const.float"
        return f"const.float({self.dtype})"


class ConstString:  # pylint: disable=too-few-public-methods
    """A string constant"""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "const.string"


class MethodSpec:
    """A spec for a compiled method"""

    method: typing.Callable
    arg_names: typing.List[str]
    arg_specs: typing.List[ArgSpecType]
    param_mode: str  # "plain", "packed", "none"
    effect_mode: str  # "plain", "packed", "none"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        method: typing.Callable,
        arg_names: typing.List[str],
        arg_specs: typing.List[ArgSpecType],
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
    def from_raw(spec: MethodSpecType, method: typing.Callable) -> "MethodSpec":
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
        config: typing.Dict[str, typing.Any] = spec.pop("$", {})  # type: ignore[assignment]
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
            if isinstance(arg_spec, (Int, Tensor)):
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
    def from_torch(args: typing.List[typing.Any], method: typing.Callable) -> "MethodSpec":
        """Converts a list of torch tensors to MethodSpec."""
        from .torch import (  # pylint: disable=import-outside-toplevel
            _method_spec_from_torch,
        )

        return _method_spec_from_torch(args, method)

    def as_inputs(self) -> typing.List[typing.Union[tir.Var, core.Tensor]]:
        """Convert the MethodSpec to a list of inputs to Module's method."""
        str2var: typing.Dict[str, tir.Var] = {}

        def _get_var(name: str) -> tir.Var:
            if name in str2var:
                return str2var[name]
            var = tir.Var(name, "int64")
            str2var[name] = var
            return var

        def _convert_input(arg_name, arg_spec):
            if isinstance(arg_spec, Int):
                arg = _get_var(arg_name)
            elif isinstance(arg_spec, Tensor):
                arg = core._tensor_placeholder(  # pylint: disable=protected-access
                    name=arg_name,
                    shape=[_get_var(x) if isinstance(x, str) else x for x in arg_spec.shape],
                    dtype=arg_spec.dtype,
                )
            elif isinstance(arg_spec, Tuple):
                elements = type(arg_spec.elements)(
                    [
                        _convert_input(
                            arg_name=arg_name + f"_tmp{i}", arg_spec=arg_spec.elements[i]
                        )
                        for i in range(len(arg_spec.elements))
                    ]
                )
                arg = Tuple(
                    name=arg_name,
                    elements=elements,
                )
            else:
                raise TypeError(f"Invalid spec for argument {arg_name}: {arg_spec}")
            return arg

        args = []
        for arg_name, arg_spec in zip(self.arg_names, self.arg_specs):
            arg = _convert_input(arg_name=arg_name, arg_spec=arg_spec)
            args.append(arg)
        return args


class ModuleSpec:
    """A spec for a compiled nn.Module"""

    module: core.Module
    method_names: typing.List[str]
    method_specs: typing.List[MethodSpec]

    def __init__(
        self,
        module: core.Module,
        method_names: typing.List[str],
        method_specs: typing.List[MethodSpec],
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
        method_specs: typing.List[MethodSpec] = []
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


class ExternFunctionSpec:  # pylint: disable=too-few-public-methods
    """A spec for a compiled external function."""

    args: typing.List[typing.Union[Tensor, ConstInt, ConstFloat, ConstString]]
    ret: typing.Union[Tensor, typing.List[Tensor]]
    symbol: typing.Optional[str]

    def __init__(
        self,
        args: typing.List[typing.Union[Tensor, ConstInt, ConstFloat, ConstString]],
        ret: typing.Union[Tensor, typing.List[Tensor]],
        symbol: typing.Optional[str] = None,
    ) -> None:
        self.args = args
        self.ret = ret
        self.symbol = symbol

    def __repr__(self) -> str:
        arg_repr = ", ".join(arg.__repr__() for arg in self.args)
        if isinstance(self.ret, list):
            ret_repr = "(" + ", ".join(ret.__repr__() for ret in self.ret) + ")"
        else:
            ret_repr = self.ret.__repr__()
        if self.symbol is None:
            func = f"({arg_repr}) -> {ret_repr}"
        else:
            func = f"{self.symbol}({arg_repr}) -> {ret_repr}"
        return f"ExternFunctionSpec: {func}"


class ExternModuleSpec:  # pylint: disable=too-few-public-methods
    """A spec for a compiled external Module."""

    library: typing.Union[str, Module]
    functions: typing.List[ExternFunctionSpec]

    def __init__(
        self,
        library: typing.Union[str, Module],
        functions: typing.List[ExternFunctionSpec],
    ) -> None:
        self.library = library
        self.functions = functions

    def load_library(self) -> Module:
        """Load the external library into Module."""
        if isinstance(self.library, Module):
            return self.library
        return load_static_library(
            self.library,
            func_names=[func.symbol for func in self.functions],
        )

    def __repr__(self) -> str:
        return f"ExternModuleSpec(library={self.library}):\n" + "\n".join(
            f"  {func.__repr__()}" for func in self.functions
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

    def build(  # pylint: disable=too-many-locals
        self, spec: ModuleSpec, debug: bool = False
    ) -> typing.Tuple[IRModule, typing.List[typing.Tuple[str, core.Parameter]]]:
        """Build the ModuleSpec to TVM IRModule. Returns the IRModule and the parameters."""

        # pylint: disable=protected-access
        def _params() -> typing.List[typing.Tuple[str, core.Parameter]]:
            params = []
            for name, param in core._attribute_finder(
                spec.module, prefix="", condition_yield=lambda x: isinstance(x, core.Parameter)
            ):
                params.append((name, param))
            return params

        def _effects() -> typing.List[typing.Tuple[str, core.Effect]]:
            result = []
            if self.io_effect is not None:
                result.append(("", self.io_effect))
            for name, effect in core._attribute_finder(
                spec.module, "", condition_yield=lambda x: isinstance(x, core.Effect)
            ):
                result.append((name, effect))
            return result

        def _extern_modules() -> typing.List[core.ExternModule]:
            result = []
            used = set()
            for _, extern_module in core._attribute_finder(
                spec.module, "", condition_yield=lambda x: isinstance(x, core.ExternModule)
            ):
                if extern_module not in used:
                    used.add(extern_module)
                    result.append(extern_module)
            return result

        # pylint: enable=protected-access

        # Disable IO effects if not in debug mode.
        if not debug:
            self.io_effect = None
        params = _params()
        effects = _effects()
        extern_modules = _extern_modules()
        with self:
            if effects:
                with self.builder.function("_initialize_effect"):
                    with self.builder.dataflow():
                        outputs = _emit_effect_init(self.builder, effects)
                    self.builder.emit_func_output(outputs, params=[])
            for method_name, method_spec in zip(spec.method_names, spec.method_specs):
                len_args = len(method_spec.arg_specs)
                len_effects = {
                    "packed": 1,
                    "none": 0,
                    "plain": len(effects),
                }[method_spec.effect_mode]
                with self.builder.function(
                    method_name,
                    attrs={"num_input": len_args + len_effects},  # type: ignore
                ):
                    with self.builder.dataflow():
                        outputs, inputs = _emit_method(self.builder, method_spec, params, effects)
                    self.builder.emit_func_output(outputs, inputs)
        external_mods = []
        for extern_module in extern_modules:
            external_mods.append(extern_module.module_spec.load_library())
        mod = self.builder.finalize()
        if extern_modules:
            original_external_mods = mod.get_attr("external_mods")
            if original_external_mods is not None:
                external_mods = original_external_mods + extern_modules
            mod = mod.with_attr("external_mods", external_mods)
        return mod, params


def _emit_effect_init(
    builder: BlockBuilder,
    effects: typing.List[typing.Tuple[str, core.Effect]],
):
    outputs = []
    for prefix, effect in effects:
        inits = effect.emit_init(prefix, builder)
        assert isinstance(inits, list)
        outputs.extend(inits)
    outputs = builder.emit_output(builder.emit(rx.Tuple(outputs)))
    return outputs


def _emit_method(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    builder: BlockBuilder,
    spec: MethodSpec,
    params: typing.List[typing.Tuple[str, core.Parameter]],
    effects: typing.Optional[typing.List[typing.Tuple[str, core.Effect]]],
):
    # pylint: disable=protected-access
    def _unwrap_ret(expr: typing.Any) -> typing.Any:
        if isinstance(expr, core.Tensor):
            return expr._expr
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
        if isinstance(arg, Tuple):
            return rx.Var(
                arg.name,
                struct_info=TupleStructInfo(
                    [_convert_input(arg_i).struct_info for arg_i in arg.elements]
                ),
            )
        raise TypeError(f"Unsupported input type: {type(arg)}")

    def _params(mode: str) -> typing.List[rx.Var]:
        inputs: typing.List[rx.Var] = []
        for name, param in params:
            var = core._tensor_placeholder(name, param.shape, param.dtype)._expr
            inputs.append(var)
            param._expr = var
        if mode == "none":
            return []
        if mode == "plain":
            return inputs
        if mode == "packed":
            input_var = rx.Var(
                "packed_params",
                TupleStructInfo(fields=[x.struct_info for x in inputs]),
            )
            for i, (name, param) in enumerate(params):
                param._expr = builder.emit(rx.TupleGetItem(input_var, i), name_hint=name)
            return [input_var]
        raise ValueError(f"Invalid param_mode: {mode}")

    def _effects(mode: str) -> typing.List[rx.Var]:
        unflat_inputs: typing.List[typing.List[rx.Var]] = []
        for name, effect in effects:
            effect_input = effect.create(name)
            effect.set_state(effect_input)
            unflat_inputs.append(effect_input)
        inputs: typing.List[rx.Var] = sum(unflat_inputs, [])
        if mode == "none":
            return []
        if mode == "plain":
            return inputs
        if mode == "packed":
            input_var = rx.Var(
                "packed_effects",
                TupleStructInfo(fields=[x.struct_info for x in inputs]),
            )
            i = 0
            for effect_input, (_, effect) in zip(unflat_inputs, effects):
                updated_effect_input = []
                for effect_input_i in effect_input:
                    updated_effect_input.append(
                        builder.emit(
                            rx.TupleGetItem(input_var, i),
                            name_hint=effect_input_i.name_hint,
                        )
                    )
                    i += 1
                effect.set_state(updated_effect_input)
            return [input_var]

        raise ValueError(f"Invalid effect_mode: {mode}")

    # pylint: enable=protected-access

    def _detuple(arg, var: rx.Var, builder: BlockBuilder):
        if isinstance(arg, Tuple):
            ret = []
            for i, elem in enumerate(arg.elements):
                field = builder.emit(rx.TupleGetItem(var, i), name_hint=f"{arg.name}_{i}")
                ret.append(_detuple(elem, field, builder))
            return type(arg.elements)(ret)
        if isinstance(arg, core.Tensor):
            return core.Tensor(_expr=var)
        if isinstance(arg, tir.Var):
            return arg
        raise TypeError(f"Unsupported input type: {type(arg)}")

    # TODO(@junrushao): Warn if params/effects are used when their mode is "none"
    explicit_inputs = spec.as_inputs()
    inputs = [_convert_input(x) for x in explicit_inputs]
    inputs = inputs + _effects(spec.effect_mode)
    inputs = inputs + _params(spec.param_mode)

    for arg_idx, (arg, var) in enumerate(zip(explicit_inputs, inputs)):
        if isinstance(arg, Tuple):
            explicit_inputs[arg_idx] = _detuple(arg, var, builder)

    outputs = spec.method(*explicit_inputs)
    effect_outputs = []
    for _, effect in effects:
        effect_outputs.extend(effect.finalize())
    if effect_outputs and spec.effect_mode != "none":
        outputs = builder.emit_output(rx.Tuple([_unwrap_ret(outputs), rx.Tuple(effect_outputs)]))
    else:
        outputs = builder.emit_output(_unwrap_ret(outputs))
    return outputs, inputs
