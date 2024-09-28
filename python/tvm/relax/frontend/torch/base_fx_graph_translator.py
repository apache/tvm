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

    def _clamp(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        a_min = args[1] if len(args) > 1 else node.kwargs["min"]
        a_max = args[2] if len(args) > 2 else node.kwargs["max"]
        if not isinstance(a_min, (int, float)):
            raise ValueError(
                f"TVM only supports constant min value for torch.clamp/clip, "
                f"but got {a_min} with type {type(a_min)}"
            )
        if not isinstance(a_max, (int, float)):
            raise ValueError(
                f"TVM only supports constant max value for torch.clamp/clip, "
                f"but got {a_max} with type {type(a_max)}"
            )
        return self.block_builder.emit(relax.op.clip(args[0], a_min, a_max))

    def _gelu(self, node: fx.Node) -> relax.Expr:
        approximate = node.kwargs.get("approximate", "none")
        if approximate == "none":
            return self.block_builder.emit(relax.op.nn.gelu(self.env[node.args[0]]))
        elif approximate == "tanh":
            return self.block_builder.emit(relax.op.nn.gelu_tanh(self.env[node.args[0]]))
        else:
            raise KeyError("Unregonized approximate algorithm for gelu: {}.".format(approximate))

    def _hardsigmoid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        return self.block_builder.emit(relax.op.divide(x1, relax.const(6, dtype)))

    def _hardswish(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        x2 = relax.op.divide(x1, relax.const(6, dtype))
        return self.block_builder.emit(relax.op.multiply(x, x2))

    def _leakyrelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("negative_slope", 0.01)
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _round(self, node: fx.Node) -> relax.Expr:
        if node.kwargs.get("decimals", 0) != 0:
            raise ValueError("specifying decimals for round is not supported yet")
        arg = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.round(arg))

    def _softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get("diagonal", 0)
            assert isinstance(k, int)
            return self.block_builder.emit(op(x, k))

        return convert

    ########## Binary Ops ##########

    def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            def promote_binary_op_args(lhs, rhs):
                if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
                    return lhs, rhs
                elif isinstance(lhs, relax.Expr):
                    assert isinstance(lhs.struct_info, relax.TensorStructInfo)
                    return lhs, relax.const(rhs, lhs.struct_info.dtype)
                elif isinstance(rhs, relax.Expr):
                    assert isinstance(rhs.struct_info, relax.TensorStructInfo)
                    return relax.const(lhs, rhs.struct_info.dtype), rhs
                else:
                    assert False

            def call_binary_op(op, lhs, rhs):
                lhs, rhs = promote_binary_op_args(lhs, rhs)
                return self.block_builder.emit(op(lhs, rhs))

            lhs, rhs = self.retrieve_args(node)
            if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
                return call_binary_op(relax_op, lhs, rhs)
            elif isinstance(lhs, relax.expr.Constant):
                return call_binary_op(relax_op, lhs, relax.const(rhs, dtype=lhs.struct_info.dtype))
            elif isinstance(rhs, relax.expr.Constant):
                return call_binary_op(relax_op, relax.const(lhs, dtype=rhs.struct_info.dtype), rhs)
            return intrinsic_op(lhs, rhs)

        return convert

    ########## Neural Network ##########

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )

    def _addmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]
        z = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.linear_algebra.matmul(y, z, out_dtype="float32"))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(bias, res))
        return res

    def _avg_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None or stride == [] else stride
        return self.block_builder.emit(
            relax.op.nn.avg_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _avg_pool2d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _baddbmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        batch1 = self.env[node.args[1]]
        batch2 = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.matmul(batch1, batch2))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(res, bias))
        return res

    def _conv_transpose1d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d_transpose = self.block_builder.emit(
            relax.op.nn.conv1d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d_transpose, bias))

    def _conv_transpose1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv_transpose2d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv2d_transpose = self.block_builder.emit(
            relax.op.nn.conv2d_transpose(
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
            return conv2d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d_transpose, bias))

    def _conv_transpose2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv1d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d = self.block_builder.emit(
            relax.op.nn.conv1d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d, bias))

    def _conv1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
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

    def _conv3d_impl(
        self,
        x: relax.Expr,
        weight: relax.Expr,
        bias: Optional[relax.Expr],
        strides: Optional[Tuple],
        padding: Optional[Tuple],
        dilation: Optional[Tuple],
        groups: Optional[Tuple],
    ):
        conv3d = self.block_builder.emit(
            relax.op.nn.conv3d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCDHW",
                kernel_layout="OIDHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv3d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv3d, bias))

    def _conv3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _einsum(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        operands = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.einsum(operands, args[0]))

    def _embedding_impl(
        self,
        x,
        weight,
    ) -> relax.Var:
        x = self.block_builder.emit(relax.op.astype(x, "int32"))

        ndim = x.struct_info.ndim
        if ndim == 1:
            return self.block_builder.emit(relax.op.take(weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = weight.struct_info.shape.values[-1]
            x = self.block_builder.emit(relax.op.reshape(x, shape=[-1]))
            embedding = self.block_builder.emit(relax.op.take(weight, x, axis=0))
            return self.block_builder.emit(relax.op.reshape(embedding, [*x_shape, emb_size]))

    def _layer_norm_impl(self, x, gamma, beta, eps, normalized_shape) -> relax.Var:
        from torch.fx.immutable_collections import immutable_list
        import numpy as np  # type: ignore

        if isinstance(normalized_shape, (immutable_list, tuple)):
            normalized_shape = tuple(normalized_shape)
        else:
            try:
                normalized_shape = self.env[normalized_shape]
            except TypeError:
                normalized_shape = tuple(normalized_shape)

        dim_num = len(normalized_shape)
        axes = list(range(-dim_num, 0))

        if gamma is None:
            shape_tuple = [int(s) for s in normalized_shape]
            gamma = relax.const(np.ones(shape_tuple), x.struct_info.dtype)
        if beta is None:
            shape_tuple = [int(s) for s in normalized_shape]
            beta = relax.const(np.zeros(shape_tuple), x.struct_info.dtype)

        return self.block_builder.emit(
            relax.op.nn.layer_norm(
                x,
                gamma,
                beta,
                axes=axes,
                epsilon=eps,
            )
        )

    def _layer_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        normalized_shape = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _layer_norm_module(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        normalized_shape = module.normalized_shape
        if module.elementwise_affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.normalized_shape), x.struct_info.dtype)
            beta = relax.const(torch.zeros_like(module.normalized_shape), x.struct_info.dtype)
        eps = module.eps
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

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

    def _scaled_dot_product_attention(self, node: fx.Node) -> relax.Var:
        transpose_S_H = lambda tensor: relax.op.permute_dims(tensor, [0, 2, 1, 3])
        query = transpose_S_H(self.env[node.args[0]])
        key = transpose_S_H(self.env[node.args[1]])
        value = transpose_S_H(self.env[node.args[2]])
        attn_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask", None)
        dropout_p = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        assert dropout_p == 0.0, "Dropout is not supported"
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        causal_mask = "TopLeft" if is_causal else None

        if attn_mask is not None:
            attn_mask = self.env[attn_mask]
            msg = "Only a float mask is supported for the attn_mask input."
            assert "float" in attn_mask.struct_info.dtype, msg

        return self.block_builder.emit(
            transpose_S_H(
                relax.op.nn.attention(query, key, value, bias=attn_mask, causal_mask=causal_mask)
            )
        )

    ########## Statistical ##########

    def _mean(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.mean(x, dim, keepdims=keepdim))

    def _sum(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        keepdim = node.kwargs["keepdim"] if "keepdim" in node.kwargs else False
        if len(args) == 1:
            return self.block_builder.emit(relax.op.sum(args[0], keepdims=keepdim))
        return self.block_builder.emit(relax.op.sum(args[0], args[1]))

    ########## Search ##########

    def _argmax_argmin(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node):
            x = self.env[node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
            return self.block_builder.emit(op(x, dim, keepdim))

        return convert

    ########## Manipulation ##########

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.reshape(x, dims))

    ########## Others ##########

    def _getitem(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.struct_info, relax.TensorStructInfo)
            take_indices = []
            take_axes = []
            stride_begin = []
            stride_end = []
            stride = []
            stride_axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            non_ellipsis_cnt = 0
            for index in node.args[1]:
                if isinstance(index, (int, slice, torch.fx.Node)):
                    non_ellipsis_cnt += 1
            for index in node.args[1]:
                if isinstance(index, int):
                    stride_begin.append(index)
                    stride_end.append(index + 1)
                    stride.append(1)
                    stride_axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    stride_begin.append(0 if index.start is None else index.start)
                    stride_end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    stride_axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(len(stride_axes) + len(expand_dim))
                elif index is Ellipsis:
                    for _ in range(len(shape) - non_ellipsis_cnt):
                        stride_begin.append(0)
                        stride_end.append(shape[i])
                        stride.append(1)
                        stride_axes.append(i)
                        i += 1
                elif isinstance(index, torch.fx.Node):
                    node_index = self.env[index]
                    if not isinstance(node_index, relax.Expr):
                        raise ValueError(
                            "Unsupported index type for relax.op.take: " + str(type(node_index))
                        )
                    take_indices.append(node_index)
                    take_axes.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                stride_begin.append(0)
                stride_end.append(shape[i])
                stride.append(1)
                stride_axes.append(i)
                i += 1
            taken = x
            if len(take_indices) > 1:
                raise ValueError("Multiple tensors as index not yet supported")
            for each_index, each_axis in zip(take_indices, take_axes):
                taken = self.block_builder.emit(relax.op.take(taken, each_index, each_axis))
            sliced = self.block_builder.emit(
                relax.op.strided_slice(taken, stride_axes, stride_begin, stride_end, stride)
            )
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        elif isinstance(x, relax.Constant):
            dtype = x.struct_info.dtype
            return relax.const(x.data.numpy()[node.args[1]], dtype)
        else:
            assert False

    @abc.abstractmethod
    def create_convert_map(
        self,
    ) -> Dict[Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]]:
        """Create convert map"""
