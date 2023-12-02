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
"""The core infra for nn.Module, which includes the following pieces:
- Tensor, a wrapper on top of relax.Expr whose struct_info is a TensorStructInfo,
  providing more convenient access shape and dtype information.
  Tensor is always symbolc and not bound to any concrete values.
- Parameter, a special tensor which could be bound or not bound to concrete values.
- Module, a container of nn.Parameters and sub nn.Modules.
- Effect, a non-user-facing class that encloses potential side effects, for example, IO,
  impure external function callings, inplace mutation, etc.
"""
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from tvm import tir
from tvm._ffi.libinfo import find_include_path
from tvm.contrib import cc as _cc
from tvm.ir import IRModule
from tvm.runtime import Device, NDArray, load_static_library, ndarray
from tvm.runtime.relax_vm import VirtualMachine
from tvm.target import Target

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo, TensorStructInfo
from ._tensor_op import _TensorOp
from .subroutine import SubroutineMixin

if TYPE_CHECKING:
    import torch

    from . import spec as _spec


_DEFAULT_DTYPE = "float32"


def get_default_dtype() -> str:
    """Get the default parameter dtype if not specified. By default it is float32.

    Returns
    -------
    dtype : str
        The default dtype
    """
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: str) -> None:
    """Set the default parameter dtype.

    Parameters
    ----------
    dtype : str
        The default dtype to be set
    """
    global _DEFAULT_DTYPE  # pylint: disable=global-statement
    _DEFAULT_DTYPE = dtype


class Tensor(_TensorOp):
    """A wrapper on top of relax.Expr whose struct_info is a TensorStructInfo, providing more
    convenient access shape and dtype information. Tensor is always symbolc and not bound to any
    concrete values. Shape and dtype inference is done eagerly upon tensor creation, i.e. when
    operators are applied on tensors, the shape and dtype information is already available.
    """

    _expr: rx.Expr

    def __init__(self, *, _expr: rx.Expr) -> None:
        """Private constructor. Tensor is never supposed to be constructed directly by users."""

        def _check_tensor(expr: rx.Expr) -> None:
            assert expr.struct_info_ is not None
            assert isinstance(expr.struct_info, TensorStructInfo)
            assert expr.struct_info.ndim != -1
            assert expr.struct_info.shape is not None
            assert expr.struct_info.shape.struct_info_ is not None
            assert isinstance(expr.struct_info.shape.struct_info, ShapeStructInfo)
            assert expr.struct_info.shape.struct_info.values is not None

        _check_tensor(_expr)
        self._expr = _expr

    @staticmethod
    def from_const(data) -> "Tensor":
        """Construct a tensor from numpy constants."""
        return Tensor(_expr=rx.const(data))

    @staticmethod
    def from_scalar(data: Union[int, float], dtype: str) -> "Tensor":
        """Construct a tensor from a scalar with dtype specified."""
        return Tensor(_expr=rx.const(data, dtype=dtype))

    @property
    def shape(self) -> List[Union[int, tir.PrimExpr]]:
        """Returns the shape of the tensor as a list of integers.

        An integer can be a python int or tvm.tir.PrimExpr, depending on whether the shape is
        fully static, for example, [1, 2, tvm.tir.Var("n")] is a valid shape where the last
        dimension is dynamic while the first two dimensions are always static constants.

        Returns
        -------
        shape : List[Union[int, tir.PrimExpr]]
            The shape of the tensor
        """

        def _simplify(expr: tir.PrimExpr):
            return expr.value if isinstance(expr, tir.IntImm) else expr

        shape_sinfo: ShapeStructInfo = self._expr.struct_info.shape.struct_info
        return [_simplify(x) for x in shape_sinfo.values]

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor.

        Returns
        -------
        ndim : int
            The number of dimensions of the tensor
        """
        return self._expr.struct_info.ndim

    @property
    def dtype(self) -> str:
        """Returns the data type of the tensor.

        Returns
        -------
        dtype : str
            The data type of the tensor
        """
        return self._expr.struct_info.dtype

    def __repr__(self) -> str:
        return f'Tensor({self.shape}, "{self.dtype}")'


class Parameter(Tensor):
    """A parameter represents the weight of a neural network layer. It is a special tensor which
    could be bound or not bound to concrete values. If a parameter is bound to a concrete value,
    it is called a bound parameter, otherwise it is called an unbound parameter.
    """

    _data: Optional[NDArray]

    def __init__(
        self,
        shape: Sequence[Union[int, tir.PrimExpr]],
        dtype: Optional[str] = None,
    ) -> None:
        """Create a parameter with given shape and dtype. The parameter is not bound to any
        concrete values.

        Parameters
        ----------
        shape : Sequence[Union[int, tir.PrimExpr]]
            The shape of the parameter
        dtype : Optional[str]
            The data type of the parameter. If not specified, the default dtype will be used.
        """
        if dtype is None:
            dtype = get_default_dtype()
        super().__init__(_expr=_tensor_placeholder("param", shape, dtype=dtype)._expr)
        self._data = None

    @property
    def data(self) -> Optional[NDArray]:
        """Returns the concrete value of the parameter if it is bound to a concrete value,
        otherwise returns None. The returned value is a tvm.runtime.NDArray."""
        return self._data

    @data.setter
    def data(self, data: Union[None, NDArray, np.ndarray, "torch.Tensor"]) -> None:
        """Set the concrete value of the parameter. The data should be one of the following:
        - None: unbind the parameter to concrete values
        - tvm.runtime.NDArray
        - numpy.ndarray
        - torch.Tensor and any other DLPack-compliant tensors
        """
        if data is None:
            self._data = data
            return
        # Try to do zero-copy if possible
        if isinstance(data, NDArray):
            pass
        elif isinstance(data, np.ndarray):
            data = ndarray.array(data)
        elif hasattr(data, "__dlpack__"):
            data = _from_dlpack(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        if data.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(self.shape)}, got {data.shape}")
        if data.dtype != self.dtype:
            raise ValueError(f"Dtype mismatch: expected {self.dtype}, got {data.dtype}")
        self._data = data

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        """Change the dtype of the parameter if it is not bound to any concrete data"""
        if dtype is not None:
            if self._data is not None:
                raise ValueError(
                    "Changing the dtype of a Parameter that has been bound to concrete "
                    "data is not recommended. It might lead to potential precision loss "
                    "or other unexpected behaviors"
                )
            self._expr = _tensor_placeholder(  # pylint: disable=protected-access
                "param", self.shape, dtype=dtype
            )._expr


class Effect:
    """Effect is a special non-user facing type that is used to represent operations with side
    effects, for example, print. It is used to represent the output of a computation.
    """

    def emit_init(self, name_hint: str, builder: BlockBuilder) -> List[rx.DataflowVar]:
        """Emit the initialization of the effect. This method is called by the compiler to
        initialize the effect."""
        raise NotImplementedError

    def create(self, name_hint: str) -> List[rx.Var]:
        """Create the implicit inputs to a relax.Function that represents the side effect"""
        raise NotImplementedError

    def set_state(self, state_vars: List[rx.Var]) -> None:
        """Set the variables that represents the effect"""
        raise NotImplementedError

    def finalize(self) -> List[rx.Var]:
        """finalize the effect as the implicit return value of a relax.Function"""
        raise NotImplementedError

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        """Convert the effect to specific dtype. Usually it is no-op for most of the effects"""


class Module(SubroutineMixin):
    """Base class for neural network components. Subclass it to build your models.
    Modules can nest within each other in a tree structure using regular attribute assignment."""

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """This method provides an iterator over module parameters,
        yielding both the parameter name and its corresponding value.

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all parameter names.

        Yields
        ------
        (str, Parameter) - Tuple containing the name and parameter
        """
        yield from _attribute_finder(
            self, prefix, condition_yield=lambda x: isinstance(x, Parameter)
        )

    def parameters(self) -> Iterator[Parameter]:
        """This method provides an iterator over module parameters,
        yielding only the Parameter value.

        Yields
        ------
        Parameter - The module's parameter
        """
        for _, param in self.named_parameters():
            yield param

    def state_dict(
        self, *, prefix: str = "", destination: Optional[Dict[str, Parameter]] = None
    ) -> Dict[str, Parameter]:
        """Returns a dictionary containing references to the whole state of the module.

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all parameter names.
        destination : Optional[Dict[str, Parameter]]
            Dictionary to which state will be saved. If None, a new dictionary is created.

        Returns
        -------
        dict : Dict[str, Parameter]
            a dictionary containing a whole state of the module
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in _attribute_finder(
            self, prefix, condition_yield=lambda x: isinstance(x, Parameter)
        ):
            destination[name] = param
        return destination

    def load_state_dict(
        self, state_dict: Dict[str, Parameter], strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        """This function copies parameters and buffers from the state_dict into the current module
        and its descendants. If `strict` is set to True, the keys in the `state_dict` must exactly
        match the keys returned by the `state_dict()` function of this module.

        Parameters
        ----------
        state_dict : Dict[str, Parameter]
            A dictionary containing a whole state of the module
        strict : bool = True
            Whether to strictly enforce that the keys in `state_dict` match the keys returned by
            this module's `state_dict()` function.

        Returns
        -------
        (missing_keys, unexpected_keys) : Tuple[List[str], List[str]]
            A tuple of two lists: the missing keys and the unexpected keys.
        """
        self_state_dict = self.state_dict()
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        for key, value in state_dict.items():
            if key not in self_state_dict:
                unexpected_keys.append(key)
                continue
            if value.data is None:
                raise ValueError(f"Parameter {key} is not set to any concrete tensor")
            self_state_dict.pop(key).data = value.data
        missing_keys = list(self_state_dict.keys())
        if strict and (missing_keys or unexpected_keys):
            raise KeyError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        return missing_keys, unexpected_keys

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module with the given inputs and returns the output."""
        if not hasattr(self, "forward"):
            raise NotImplementedError(f"Module {type(self)} does not have a `forward` method")
        return self.forward(*args, **kwargs)  # pylint: disable=no-member

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        """Convert the module to specific dtype recursively"""
        for _, item in self.__dict__.items():
            if hasattr(item, "to") and callable(item.to):
                item.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init

    def export_tvm(
        self,
        spec: "_spec.ModuleSpecType",
        debug: bool = False,
    ) -> Tuple[IRModule, List[Tuple[str, Parameter]]]:
        """Export the module to TVM IRModule and parameters

        Parameters
        ----------
        spec : _spec.ModuleSpecType
            A dictionary mapping each input name to a specification
            that defines the inputs shape and dtype.
        debug : bool
            If set to True, then the exported module will support
            effects. This enables things like printing in the graph.

        Returns
        -------
        irmodule : tvm.ir.IRModule
            The converted tvm IR representation of the model.
        params : Dict[str, tvm.nd.array]
            A dictionary of parameters corresponding to the weights of
            the model.
        """
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec, debug=debug)
        return mod, params

    def jit(  # pylint: disable=too-many-arguments
        self,
        spec: "_spec.ModuleSpec",
        target: Union[str, Target] = "llvm",
        device: Union[str, Device] = "cpu",
        pipeline: str = "zero",
        out_format: str = "torch",
        debug: bool = False,
    ) -> Any:
        """Just-in-time compilation of a nn.model to an executable"""
        from tvm import relax  # pylint: disable=import-outside-toplevel

        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        # Convert nn.Module to IRModule
        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec, debug=debug)

        # Convert parameters
        device = _str_to_device(device)
        params_ndarray = _param_to_ndarray(params, device)

        # Compile mod and feed it to VM
        mod = relax.pipeline.get_pipeline(pipeline)(mod)  # pylint: disable=no-value-for-parameter
        vm = VirtualMachine(  # pylint: disable=invalid-name
            relax.build(mod, target=target),
            device,
        )

        if out_format == "torch":
            from . import torch  # pylint: disable=import-outside-toplevel

            return torch.TorchModule(spec=spec, params=params_ndarray, vm=vm)

        raise ValueError(f"Unknown out_format: {out_format}")


class ExternModule(Module):
    """Base class for external module. Subclass it to import your external models.
    Modules can nest within each other in a tree structure using regular attribute assignment."""

    module_spec: "_spec.ExternModuleSpec"

    def __init__(self, module_spec: "_spec.ExternModuleSpec") -> None:
        super().__init__()
        self.module_spec = module_spec

    def get_extern_func(self, func_name: str) -> Callable:
        """This method helps get the external funciton in external module by function name.
        It will wrap the functions as other prebuilt operators.

        Parameters
        ----------
        func_name : str
            The name of the function to get.

        Returns
        ------
        ret_func: Callable
            The callable function to call.
        """
        for function_spec in self.module_spec.functions:
            if function_spec.symbol == func_name:
                # pylint: disable=cell-var-from-loop, import-outside-toplevel, protected-access
                from tvm.relax import Tuple as RxTuple
                from tvm.relax import call_dps_packed

                from . import spec as _spec
                from .op import _wrap_nested

                def extern_func(
                    *args: List[
                        Union[_spec.Tensor, _spec.ConstInt, _spec.ConstFloat, _spec.ConstString]
                    ]
                ) -> Tensor:
                    spec2var = {}
                    for arg, arg_spec in zip(args, function_spec.args):
                        if not isinstance(arg_spec, _spec.Tensor):
                            continue
                        for value, value_spec in zip(arg.shape, arg_spec.shape):
                            if isinstance(value_spec, str):
                                if not value_spec in spec2var:
                                    spec2var[value_spec] = value
                                else:
                                    if not spec2var[value_spec] == value:
                                        raise ValueError(
                                            f"Confilict vars {spec2var[value_spec]} and {value} "
                                            f"for {value_spec} in {function_spec}"
                                        )
                    out_shape = []
                    func_spec_ret = function_spec.ret
                    assert isinstance(
                        func_spec_ret, _spec.Tensor
                    ), "Only single return value is supported for now"
                    for value_spec in func_spec_ret.shape:
                        if isinstance(value_spec, int):
                            out_shape.append(value_spec)
                        elif isinstance(value_spec, str):
                            if not value_spec in spec2var:
                                raise ValueError(f"Undefined var {value_spec} in {function_spec}")
                            out_shape.append(spec2var[value_spec])
                    out_sinfo = TensorStructInfo(
                        out_shape,  # type: ignore[arg-type]
                        func_spec_ret.dtype,
                    )
                    relax_args = []
                    for arg, arg_spec in zip(args, function_spec.args):
                        if isinstance(arg_spec, _spec.Tensor):
                            relax_args.append(arg._expr)
                        elif isinstance(arg_spec, _spec.ConstInt):
                            if arg_spec.dtype is None:
                                relax_args.append(rx.PrimValue(int(arg)))
                            else:
                                relax_args.append(rx.PrimValue(tir.IntImm(arg_spec.dtype, arg)))
                        elif isinstance(arg_spec, _spec.ConstFloat):
                            if arg_spec.dtype is None:
                                relax_args.append(rx.PrimValue(float(arg)))
                            else:
                                relax_args.append(rx.PrimValue(tir.FloatImm(arg_spec.dtype, arg)))
                        elif isinstance(arg_spec, _spec.ConstString):
                            relax_args.append(rx.StringImm(arg))

                    ret_tensor = _wrap_nested(
                        call_dps_packed(
                            func_name,
                            args=RxTuple(relax_args),
                            out_sinfo=out_sinfo,
                        ),
                        func_name,
                    )
                    assert isinstance(ret_tensor, Tensor)
                    return ret_tensor

                # pylint: enable=cell-var-from-loop, import-outside-toplevel, protected-access

                return extern_func
        raise ValueError(f"Unknown function {func_name} in the external module:{self.module_spec}")


class SourceModule(ExternModule):
    """Base class for source module. Subclass it to import your source models.

    See PR #16006 (https://github.com/apache/tvm/pull/16006) for a detailed example.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        source_code: str,
        source_format: str,  # "cpp", "cu"
        functions: Dict[str, "_spec.ExternFunctionSpec"],
        compile_options: Optional[List[str]] = None,
        compiler: Optional[str] = None,
        output_format: str = "obj",  # "obj", "wasm"
    ):
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        def _detect_input_suffix(source_format: str) -> str:
            if source_format == "cpp":
                return ".cpp"
            if source_format == "cu":
                return ".cu"
            raise ValueError(f"Invalid source format: {source_format}")

        def _detect_output_suffix(output_format: str) -> str:
            if output_format == "obj":
                if _cc._is_linux_like():  # pylint: disable=protected-access
                    return ".o"
                if _cc._is_windows_like():  # pylint: disable=protected-access
                    return ".obj"
                raise ValueError(f"Unsupported platform: {sys.platform}")
            if output_format == "wasm":
                return ".wasm"
            raise ValueError(f"Invalid output format: {output_format}")

        source_suffix = _detect_input_suffix(source_format)
        output_suffix = _detect_output_suffix(output_format)
        if compile_options is None:
            compile_options = []
            for include_path in find_include_path():
                compile_options.append("-I")
                compile_options.append(include_path)
            compile_options.append("-c")
            compile_options.append("-DDMLC_USE_FOPEN64=0")
            compile_options.append("-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>")
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = f"main{source_suffix}"
            with open(
                os.path.join(temp_dir, f"main{source_suffix}"), "w", encoding="utf-8"
            ) as file:
                file.write(source_code)
            output_file = f"main{output_suffix}"
            if shutil.which("ccache"):
                ccache_env = {"CCACHE_COMPILERCHECK": "content"}
            else:
                ccache_env = None
            _cc.create_shared(
                output=output_file,
                objects=[source_file],
                options=compile_options,
                cc=compiler,
                cwd=temp_dir,
                ccache_env=ccache_env,
            )
            func_names: List[str] = []
            func_specs: List[_spec.ExternFunctionSpec] = []
            for func_name, func_spec in functions.items():
                func_names.append(func_name)
                func_specs.append(func_spec)
                if func_spec.symbol is None:
                    func_spec.symbol = func_name
            library = load_static_library(
                os.path.join(temp_dir, f"main{output_suffix}"), func_names=func_names
            )
        module_spec = _spec.ExternModuleSpec(
            library=library,
            functions=func_specs,
        )
        super().__init__(module_spec=module_spec)


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: List[Module]):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __setitem__(self, idx, module):
        self.modules[idx] = module

    def __len__(self):
        return len(self.modules)

    def append(self, module):
        """Add a module to the end of the ModuleList"""
        self.modules.append(module)

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        for module in self.modules:
            module.to(dtype=dtype)

    def forward(self, x):  # pylint: disable=invalid-name
        """Feed-forward pass of the module"""
        for module in self.modules:
            x = module(x)
        return x


def _attribute_finder(root: Module, prefix: str, condition_yield: Callable[[Any], bool]):
    """Find attributes that satisfy the condition recursively"""
    for name, item in root.__dict__.items():
        if condition_yield(item):
            yield prefix + name, item
        elif isinstance(item, ModuleList):
            for i, subitem in enumerate(item):
                yield from _attribute_finder(
                    subitem,
                    prefix + name + f".{i}.",
                    condition_yield,
                )
        elif isinstance(item, Module):
            yield from _attribute_finder(
                item,
                prefix + name + ".",
                condition_yield,
            )


def _tensor_placeholder(
    name: str, shape: Sequence[Union[int, tir.PrimExpr]], dtype: str
) -> "Tensor":
    new_shape = []
    for expr in shape:
        if isinstance(expr, (int, tir.IntImm)):
            expr = int(expr)
            assert expr >= 0
            new_shape.append(expr)
            continue
        if not isinstance(expr, tir.PrimExpr):
            raise TypeError(f"Invalid shape: {shape}")
        assert expr.dtype == "int64"
        new_shape.append(expr)
    return Tensor(
        _expr=rx.Var(
            name_hint=name,
            struct_info=TensorStructInfo(
                shape=new_shape,  # type: ignore[arg-type]
                dtype=dtype,
            ),
        )
    )


def _from_dlpack(tensor) -> NDArray:
    try:
        return ndarray.from_dlpack(tensor)
    except RuntimeError:
        pass
    # special logic for PyTorch
    device_type = tensor.device.type
    device_id = tensor.device.index or 0
    return ndarray.array(
        tensor.numpy(),
        device=Device(
            Device.STR2MASK[device_type],
            device_id,
        ),
    )


def _str_to_device(device: Union[str, Device]) -> Device:
    if isinstance(device, Device):
        return device
    split = device.split(":")
    if len(split) > 2:
        raise ValueError(f"Invalid device: {device}")
    device_type = split[0]
    device_id = 0 if len(split) == 1 else int(split[1])
    if device_type not in Device.STR2MASK:
        raise ValueError(f"Unsupported device type: {device_type}")
    return Device(Device.STR2MASK[device_type], device_id)


def _param_to_ndarray(params: List[Tuple[str, Parameter]], device: Device) -> List[NDArray]:
    results = []
    missing = []
    for name, param in params:
        if param.data is None:
            missing.append(name)
        else:
            results.append(param.data.copyto(target=device))
    if missing:
        raise ValueError(f"Parameters are not set to any concrete values: {', '.join(missing)}")
    return results
