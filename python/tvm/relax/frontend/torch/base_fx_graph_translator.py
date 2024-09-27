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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""Base class for PyTorch FX Graph importer."""
import abc
from typing import Callable, Dict, Optional, Tuple, Union

from tvm import relax


class BaseFXGraphImporter(metaclass=abc.ABCMeta):
    """Base class for FX Graph Importer."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore
        from torch import fx

        self.env: Dict[fx.Node, relax.Expr] = {}
        self.params: Dict[torch.Tensor, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.convert_map: Dict[
            Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]
        ] = self.create_convert_map()

    ########## Utilities ##########

    @staticmethod
    def _convert_data_type(input_type: Union[str, torch.dtype], env: Optional[Dict] = None):
        """converts the PyTorch scalar type input_type to a TVM dtype."""
        import torch  # type: ignore

        if env is not None and input_type in env:
            input_type = env[input_type]

        input_type = input_type.lower() if isinstance(input_type, str) else input_type
        if input_type in ["float", "float32", "torch.float32", torch.float32]:
            return "float32"
        elif input_type in ["float16", "torch.float16", torch.float16]:
            return "float16"
        elif input_type in ["int64", "torch.int64", torch.int64]:
            return "int64"
        elif input_type in ["int32", "torch.int32", torch.int32]:
            return "int32"
        elif input_type in ["bool", "torch.bool", torch.bool]:
            return "bool"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        tensor = tensor.detach().cpu()
        dtype = BaseFXGraphImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), dtype)

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""
        import torch  # type: ignore

        if isinstance(tensor, relax.Expr):
            if not isinstance(tensor.struct_info, relax.TensorStructInfo):
                raise TypeError("The input Expr of shape_of should be a Tensor")
            return tensor.struct_info.shape
        elif isinstance(tensor, torch.Tensor):
            return tensor.shape
        raise ValueError("Unsupported type: {}".format(type(tensor)))

    def retrieve_args(self, node: fx.Node):
        return self._retrieve_args(node.args)

    def _retrieve_args(self, node):
        from torch import fx

        if isinstance(node, fx.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrieve_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()}
        else:
            return node

    ########## Unary Ops ##########

    def _unary_op(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            return self.block_builder.emit(op(self.env[node.args[0]]))

        return convert

    ########## Neural Network ##########

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )

    def _conv2d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ):
        conv2d = self.block_builder.emit(
            relax.op.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv2d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d, bias))

    def _conv2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _linear(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None else stride
        return self.block_builder.emit(
            relax.op.nn.max_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _max_pool2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    ########## Manipulation ##########

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.reshape(x, dims))

    ########## Others ##########

    @abc.abstractmethod
    def create_convert_map(
        self,
    ) -> Dict[Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]]:
        """Create convert map"""
