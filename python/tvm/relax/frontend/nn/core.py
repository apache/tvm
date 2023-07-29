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
from tvm.ir import IRModule
from tvm.runtime import Device, NDArray, ndarray
from tvm.runtime.relax_vm import VirtualMachine
from tvm.target import Target

from ... import expr as rx
from ...block_builder import BlockBuilder
from ...struct_info import ShapeStructInfo, TensorStructInfo
from ._tensor_op import _TensorOp

if TYPE_CHECKING:
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

    def __init__(self, shape: List[Union[int, tir.PrimExpr]], dtype: Optional[str] = None) -> None:
        """Create a parameter with given shape and dtype. The parameter is not bound to any
        concrete values.

        Parameters
        ----------
        shape : List[Union[int, tir.PrimExpr]]
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

    def finalize(self) -> List[rx.Var]:
        """finalize the effect as the implicit return value of a relax.Function"""
        raise NotImplementedError

    def to(self, dtype: Optional[str] = None) -> None:  # pylint: disable=invalid-name
        """Convert the effect to specific dtype. Usually it is no-op for most of the effects"""


class Module:
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

    def export_tvm(
        self,
        spec: "_spec.ModuleSpecType",
    ) -> Tuple[IRModule, List[Tuple[str, Parameter]]]:
        """Export the module to TVM IRModule and parameters"""
        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec)
        return mod, params

    def jit(  # pylint: disable=too-many-arguments
        self,
        spec: "_spec.Module",
        target: Union[str, Target] = "llvm",
        device: str = "cpu",
        pipeline: str = "zero",
        out_format: str = "torch",
    ) -> Callable:
        """Just-in-time compilation of a nn.model to an executable"""
        from tvm import relax  # pylint: disable=import-outside-toplevel

        from . import spec as _spec  # pylint: disable=import-outside-toplevel

        # Convert nn.Module to IRModule
        spec = _spec.ModuleSpec.from_raw(spec, self)
        mod, params = _spec.SpecBuilder().build(spec)

        # Convert parameters
        device = _str_to_device(device)
        params = _param_to_ndarray(params, device)

        # Compile mod and feed it to VM
        mod = relax.pipeline.get_pipeline(pipeline)(mod)  # pylint: disable=no-value-for-parameter
        mod = relax.build(mod, target=target)
        vm = VirtualMachine(mod, device)  # pylint: disable=invalid-name

        if out_format == "torch":
            from . import torch  # pylint: disable=import-outside-toplevel

            return torch.TorchModule(spec=spec, params=params, vm=vm)

        raise ValueError(f"Unknown out_format: {out_format}")


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
                shape=new_shape,
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


def _str_to_device(device: str) -> Device:
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
