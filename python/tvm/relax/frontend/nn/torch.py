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
"""PyTorch integration with nn.Module"""
import inspect
from typing import Any, Callable, List

import torch

from tvm.ir import Array
from tvm.runtime import NDArray, ShapeTuple, ndarray
from tvm.runtime.relax_vm import VirtualMachine

from . import core
from . import spec as _spec


class TorchModule:  # pylint: disable=too-few-public-methods
    """A wrapper on top of TVM VirtualMachine that takes torch tensors as inputs and returns torch
    tensors as outputs"""

    spec: _spec.ModuleSpec
    vm: VirtualMachine  # pylint: disable=invalid-name
    params: List[NDArray]
    effects: List[Any]

    def __init__(  # pylint: disable=invalid-name
        self,
        spec: _spec.ModuleSpec,
        vm: VirtualMachine,
        params: List[NDArray],
    ):
        try:
            self.effects = vm["_initialize_effect"]()
        except AttributeError:
            self.effects = None

        self.spec = spec
        self.vm = vm
        self.params = params

    def __getitem__(self, method_name: str) -> Callable:
        def _find_method(method_name):
            for key, value in zip(self.spec.method_names, self.spec.method_specs):
                if method_name == key:
                    return value
            raise ValueError(f"Method `{method_name}` is not found in the module spec. {self.spec}")

        method_spec = _find_method(method_name)
        method = self.vm[method_name]

        def _closure(*args):
            if len(args) != len(method_spec.arg_names):
                raise TypeError(
                    f"Argument length mismatch. Expected {len(method_spec.arg_names)} arguments, "
                    f"but got {len(args)} arguments. The spec is: {method_spec}"
                )
            args = [
                _torch_to_tvm(arg_name, arg_spec, arg)
                for arg_name, arg_spec, arg in zip(
                    method_spec.arg_names, method_spec.arg_specs, args
                )
            ]
            if self.effects is not None:
                outputs, self.effects = method(*args, *self.effects, *self.params)
            else:
                outputs = method(*args, *self.params)
            return _tvm_to_torch(outputs)

        _closure.__name__ = method_name
        return _closure


def _tvm_to_torch(arg):
    if isinstance(arg, (list, tuple, Array)):
        return [_tvm_to_torch(i) for i in arg]
    if isinstance(arg, ndarray.NDArray):
        return torch.utils.dlpack.from_dlpack(arg)
    if isinstance(arg, ShapeTuple):
        return list(arg)
    raise TypeError(f"Unsupported argument type: {type(arg)}")


def _torch_to_tvm(arg_name, arg_spec, arg_torch):
    if isinstance(arg_spec, _spec.Tensor):
        if not isinstance(arg_torch, torch.Tensor):
            raise TypeError(
                f"Expected argument `{arg_name}` to be `torch.Tensor`, "
                f"but got {type(arg_torch)}"
            )
        return core._from_dlpack(arg_torch)  # pylint: disable=protected-access
    if isinstance(arg_spec, _spec.Int):
        if not isinstance(arg_torch, int):
            raise TypeError(
                f"Expected argument `{arg_name}` to be `int`, but got {type(arg_torch)}"
            )
        return ShapeTuple([arg_torch])
    if isinstance(arg_spec, _spec.Tuple):
        return [
            _torch_to_tvm(f"{arg_name}[{i}]", x, arg_torch[i])
            for i, x in enumerate(arg_spec.elements)
        ]
    raise TypeError(f"Unsupported spec item type: {type(arg_spec)}")


def _method_spec_from_torch(
    args_torch: List[Any],
    method: Callable,
):
    def _as_spec(arg_torch):
        if isinstance(arg_torch, torch.Tensor):
            _, dtype = str(arg_torch.dtype).rsplit(".", maxsplit=1)
            return _spec.Tensor(shape=list(arg_torch.shape), dtype=dtype)
        if isinstance(arg_torch, int):
            return _spec.Int()
        raise TypeError(f"Unsupported argument type: {type(arg_torch)}")

    arg_names = list(inspect.signature(method).parameters.keys())
    if len(arg_names) != len(args_torch):
        raise TypeError(f"Expected {len(arg_names)} arguments, but got {len(args_torch)} arguments")
    arg_specs = [_as_spec(i) for i in args_torch]
    return _spec.MethodSpec(method, arg_names, arg_specs, param_mode="plain", effect_mode="plain")
