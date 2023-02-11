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
# pylint: disable=redefined-builtin
"""PyTorch-like nn.Module API for constructing workloads."""


from typing import List, Any, Callable, Union
import typing
import numpy as np  # type: ignore

import tvm
from tvm import relax, topi, tir


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> relax.Var:
    return relax.BlockBuilder.current().emit_te(func, *args, **kwargs)


class Placeholder(relax.Var):
    """A placeholder variable that can represent model input."""

    def __init__(
        self, shape: Union[List[Any], typing.Tuple[Any, ...]], dtype="float32", name="data"
    ):
        if not isinstance(shape, (list, tuple)):
            raise TypeError("the shape of Placeholder is expected to be a list or a tuple")
        super().__init__(
            relax.BlockBuilder.current().get_unique_name(name), relax.TensorStructInfo(shape, dtype)
        )


class Parameter(relax.Var):
    """A special kind of relax Var that represents model parameter(weight)."""

    def __init__(
        self, shape: Union[List[Any], typing.Tuple[Any, ...]], dtype="float32", name="param"
    ):
        if not isinstance(shape, (list, tuple)):
            raise TypeError("the shape of Parameter is expected to be a list or a tuple")
        super().__init__(
            relax.BlockBuilder.current().get_unique_name(name), relax.TensorStructInfo(shape, dtype)
        )


class Module:
    """Base class for all model modules.

    A neural network or a layer can subclass this class.

    Example
    -------
    .. code-block:: python

        # Define a linear layer
        class Linear(Module)
            def __init__(self, in_features, out_features, bias=True):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = Parameter((in_features, out_features), name="linear_weight")
                if bias:
                    self.bias = Parameter((out_features,), name="linear_bias")
                else:
                    self.bias = None

            # All submodules should implement forward.
            # Defines the forward computation performed at every call.
            def forward(self, input: relax.Expr) -> relax.Var:
                y = emit_te(topi.matmul, input, self.weight)
                if self.bias is not None:
                    y = emit_te(topi.add, y, self.bias)
                return y
    """

    def parameters(self) -> List[Parameter]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def forward(self, input: relax.Expr):
        """Define the computation performed at every call."""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _unpack_params(value: object) -> List[relax.Var]:
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        params = []
        for v in value.values():
            params += _unpack_params(v)
        return params
    if isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    if value is None or isinstance(value, (int, float, str)):
        return []
    raise TypeError("not supported type when unpacking parameters: {}".format(type(value)))


def init_params(mod: tvm.IRModule) -> List[tvm.nd.array]:
    """Utility function to initialize model's parameters."""
    shape_dict = {v.name_hint: v.struct_info.shape for v in mod["main"].params}
    params = []
    for k, v in shape_dict.items():
        if k.startswith("data"):
            continue
        if isinstance(v, relax.ShapeExpr):
            shape = []
            for i in v:
                if isinstance(i, tir.IntImm):
                    shape.append(int(i))
                else:
                    raise TypeError("cannot initialize for unknown-shape parameters.")
            params.append(tvm.nd.array(np.zeros(shape).astype(np.float32)))
        else:
            raise TypeError("cannot initialize for unknown-shape parameters.")
    return params


class Sequential(Module):
    """A sequential container that concatenates modules in it.

    Example
    -------
    .. code-block:: python

        model = nn.Sequential(
                    nn.Conv2d(1, 20, 5),
                    nn.ReLU(),
                    nn.Conv2d(20, 64, 5),
                    nn.ReLU()
                )
    """

    def __init__(self, *modules: Module):
        self.modules = modules

    def forward(self, input: relax.Expr) -> relax.Var:
        for module in self.modules:
            input = module(input)
        return input


class ReLU(Module):
    """Applies the rectified linear unit activation function on the input."""

    def forward(self, input: relax.Expr) -> relax.Var:
        return emit_te(topi.nn.relu, input)


class LogSoftmax(Module):
    """Applies log softmax activation function on the input."""

    def forward(self, input: relax.Expr) -> relax.Var:
        return emit_te(topi.nn.log_softmax, input)


class Linear(Module):
    """Applies a linear transformation to the input data: :math:`y = xA + b`."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((in_features, out_features), name="linear_weight")
        if bias:
            self.bias = Parameter((out_features,), name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        y = emit_te(topi.matmul, input, self.weight)
        if self.bias is not None:
            y = emit_te(topi.add, y, self.bias)
        return y
