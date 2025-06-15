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
from functools import reduce
import math
from typing import Callable, Dict, Optional, Tuple, Union, List

from tvm import relax, tir


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
        elif input_type in ["bfloat16", "torch.bfloat16", torch.bfloat16]:
            return "bfloat16"
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

    def _check_unsupported_func_type(self, nodes: List[fx.Node]):
        missing_func_types = list(
            {
                node.target.__name__
                for node in nodes
                if node.op == "call_function" and node.target.__name__ not in self.convert_map
            }
        )
        assert not missing_func_types, f"Unsupported function types {missing_func_types}"

    ########## Unary Ops ##########

    def _unary_op(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            return self.block_builder.emit(op(self.env[node.args[0]]))

        return convert

    def _celu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.struct_info.dtype

        if isinstance(alpha, (int, float)):
            alpha = relax.const(alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(alpha, dtype))

        zero = relax.const(0, dtype)
        # alpha * min(0, exp(x / alpha) - 1) + max(0, x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.minimum(
                        zero,
                        relax.op.subtract(
                            relax.op.divide(relax.op.exp(x), alpha), relax.const(1, dtype)
                        ),
                    ),
                ),
                relax.op.nn.relu(x),
            )
        )

    def _clamp(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = args[1] if len(args) > 1 else node.kwargs.get("min", -math.inf)
        a_max = args[2] if len(args) > 2 else node.kwargs.get("max", math.inf)

        a_min = -math.inf if a_min is None else a_min
        a_max = math.inf if a_max is None else a_max

        # Handle the case where a_min is a tensor
        if not isinstance(a_min, (int, float)):
            from torch import fx

            if isinstance(a_min, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_min = self.env[a_min]
            assert isinstance(a_min, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_min} with type {type(a_min)}"
            )
            a_min = self.block_builder.emit(relax.op.broadcast_to(a_min, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.maximum(x, a_min))
            a_min = -math.inf

        # Handle the case where a_max is a tensor
        if not isinstance(a_max, (int, float)):
            from torch import fx

            if isinstance(a_max, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_max = self.env[a_max]
            assert isinstance(a_max, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_max} with type {type(a_max)}"
            )
            a_max = self.block_builder.emit(relax.op.broadcast_to(a_max, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.minimum(x, a_max))
            a_max = math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _clamp_min(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = args[1] if len(args) > 1 else node.kwargs.get("min", -math.inf)
        a_max = math.inf

        a_min = -math.inf if a_min is None else a_min

        # Handle the case where a_min is a tensor
        if not isinstance(a_min, (int, float)):
            from torch import fx

            if isinstance(a_min, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_min = self.env[a_min]
            assert isinstance(a_min, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_min} with type {type(a_min)}"
            )
            a_min = self.block_builder.emit(relax.op.broadcast_to(a_min, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.maximum(x, a_min))
            a_min = -math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _clamp_max(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        a_min = -math.inf
        a_max = args[2] if len(args) > 2 else node.kwargs.get("max", math.inf)

        a_max = math.inf if a_max is None else a_max

        # Handle the case where a_max is a tensor
        if not isinstance(a_max, (int, float)):
            from torch import fx

            if isinstance(a_max, fx.Node):
                # Extract relax Expr (needed for fx.tracer)
                a_max = self.env[a_max]
            assert isinstance(a_max, relax.Expr), (
                f"Unexpected argument type "
                f"passed to torch.clamp/clip: {a_max} with type {type(a_max)}"
            )
            a_max = self.block_builder.emit(relax.op.broadcast_to(a_max, self.shape_of(x)))
            x = self.block_builder.emit(relax.op.minimum(x, a_max))
            a_max = math.inf

        return self.block_builder.emit(relax.op.clip(x, a_min, a_max))

    def _elu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.struct_info.dtype

        if isinstance(alpha, (int, float)):
            alpha = relax.const(-alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(-alpha, dtype))

        # alpha * ReLU(1 − exp(x)) + ReLU(x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.nn.relu(relax.op.subtract(relax.const(1, dtype), relax.op.exp(x))),
                ),
                relax.op.nn.relu(x),
            )
        )

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

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.kwargs.get("min_val", -1.0)
        max_val = node.kwargs.get("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    def _leakyrelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("negative_slope", 0.01)
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _prelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = self.env[node.args[1]]
        axis = 0 if len(x.struct_info.shape) == 1 else 1
        return self.block_builder.emit(relax.op.nn.prelu(x, alpha, axis))

    def _round(self, node: fx.Node) -> relax.Expr:
        if node.kwargs.get("decimals", 0) != 0:
            raise ValueError("specifying decimals for round is not supported yet")
        arg = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.round(arg))

    def _softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _softplus(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        beta = node.args[1] if len(node.args) > 1 else node.kwargs.get("beta", 1.0)
        threshold = node.args[2] if len(node.args) > 2 else node.kwargs.get("threshold", 20.0)
        return self.block_builder.emit(relax.op.nn.softplus(x, beta, threshold))

    def _softshrink(self, node: fx.Node) -> relax.Var:
        """
        Applies the Softshrink activation function in Relax.

        Softshrink(x) =
            x - λ    if x > λ
            x + λ    if x < -λ
            0        otherwise

        Args:
            node (fx.Node): The input node containing the tensor and lambda value.

        Returns:
            relax.Var: The resulting tensor after applying Softshrink.
        """
        args = self.retrieve_args(node)
        x = args[0]
        lambd = relax.const(args[1] if len(args) > 1 else 0.5, x.struct_info.dtype)

        # Apply Softshrink transformation with masking
        shrink_pos = relax.op.multiply(
            relax.op.subtract(x, lambd),
            relax.op.astype(relax.op.greater(x, lambd), x.struct_info.dtype),
        )

        shrink_neg = relax.op.multiply(
            relax.op.add(x, lambd),
            relax.op.astype(relax.op.less(x, relax.op.negative(lambd)), x.struct_info.dtype),
        )

        # Combine the positive and negative shrink results
        return self.block_builder.emit(relax.op.add(shrink_pos, shrink_neg))

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

    def _div(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        inp_1 = args[0]
        inp_2 = args[1]

        # Handle scalar cases
        if isinstance(inp_2, (int, float)):
            inp_2 = relax.const(inp_2)

        # Get rounding_mode from node kwargs
        rounding_mode = args[2] if len(node.args) > 2 else node.kwargs.get("rounding_mode", None)

        # Perform division based on rounding mode
        if rounding_mode is None:
            # True division (normal float division)
            return self.block_builder.emit(relax.op.divide(inp_1, inp_2))
        elif rounding_mode == "floor":
            # Floor division
            return self.block_builder.emit(relax.op.floor_divide(inp_1, inp_2))
        elif rounding_mode == "trunc":
            # Trunc division: perform true division then truncate
            true_div = self.block_builder.emit(relax.op.divide(inp_1, inp_2))
            return self.block_builder.emit(relax.op.trunc(true_div))
        else:
            raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")

    def _fmod(self, node: fx.Node):
        args = self.retrieve_args(node)
        lhs = args[0]
        rhs = args[1]
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return self.block_builder.emit(relax.op.mod(lhs, rhs))
        elif isinstance(lhs, relax.Expr):
            rhs = relax.const(rhs, lhs.struct_info.dtype)
        elif isinstance(rhs, relax.Expr):
            lhs = relax.const(lhs, rhs.struct_info.dtype)
        else:
            assert False
        return self.block_builder.emit(relax.op.mod(lhs, rhs))

    def _rsub(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        lhs = args[0]
        rhs = args[1]

        if isinstance(rhs, (int, float)):
            rhs = relax.const(rhs)

        return self.block_builder.emit(relax.op.subtract(rhs, lhs))

    def _isin(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        elements = args[0]
        test_elements = args[1]

        expanded_elements = relax.op.expand_dims(elements, axis=-1)
        flattened_test_elements = relax.op.reshape(test_elements, (-1,))

        comparison = relax.op.equal(expanded_elements, flattened_test_elements)
        summed = relax.op.sum(comparison, axis=-1)
        result = relax.op.greater(summed, relax.const(0, dtype=elements.struct_info.dtype))

        return self.block_builder.emit(result)

    ########## Linear Algebra ##########

    def _linalg_vector_norm(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)

        data = args[0]
        # Default ord=2 if not supplied
        ord_val = args[1] if len(args) > 1 else 2.0
        dim = args[2] if len(args) > 2 else None
        keepdim = args[3] if len(args) > 3 else False

        # If ord_val is a Python float/int, wrap it in a Relax const
        # so that it matches data's dtype.
        dtype = data.struct_info.dtype
        ord_expr = (
            ord_val if isinstance(ord_val, relax.Expr) else relax.const(float(ord_val), dtype)
        )
        # Reciprocal
        reci_expr = (
            relax.op.divide(relax.const(1.0, dtype), ord_expr)
            if isinstance(ord_val, relax.Expr)
            else relax.const(1.0 / float(ord_val), dtype)
        )

        # abs(data)
        abs_data = self.block_builder.emit(relax.op.abs(data))
        # abs_data^ord
        abs_data_pow = self.block_builder.emit(relax.op.power(abs_data, ord_expr))
        # sum over dim
        reduced = self.block_builder.emit(relax.op.sum(abs_data_pow, dim, keepdims=keepdim))
        # (sum(...))^(1/ord)
        norm_val = self.block_builder.emit(relax.op.power(reduced, reci_expr))

        return norm_val

    ########## Neural Network ##########

    def _adaptive_avg_pool1d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1] if len(node.args) > 1 else node.kwargs["output_size"]
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.struct_info.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool1d(x, output_size, layout="NCW")
        )
        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.struct_info.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )
        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _adaptive_avg_pool3d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.struct_info.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)

        result = self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool3d(x, output_size, layout="NCDHW")
        )
        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

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

    def _avg_pool1d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int]] = 1,
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Optional[int] = 0,
        ceil_mode: Optional[bool] = False,
        count_include_pad: Optional[bool] = True,
    ) -> relax.Var:
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.struct_info.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool1d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout="NCW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool1d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)

        return self._avg_pool1d_impl(x, kernel_size, stride, padding, ceil_mode, count_include_pad)

    def _avg_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.struct_info.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool2d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _avg_pool3d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[int] = 0,
        ceil_mode: Optional[bool] = False,
        count_include_pad: Optional[bool] = True,
    ) -> relax.Var:
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.struct_info.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)
        stride = kernel_size if stride is None or stride == [] else stride

        result = self.block_builder.emit(
            relax.op.nn.avg_pool3d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout="NCDHW",
            )
        )
        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _avg_pool3d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)

        return self._avg_pool3d_impl(x, kernel_size, stride, padding, ceil_mode, count_include_pad)

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
        output_padding: Optional[Tuple],
    ) -> relax.Var:
        conv1d_transpose = self.block_builder.emit(
            relax.op.nn.conv1d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                output_padding=output_padding,
                data_layout="NCW",
                kernel_layout="IOW",
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
        output_padding = args[5] if len(args) > 5 else 0
        groups = args[6] if len(args) > 6 else 1
        dilation = args[7] if len(args) > 7 else 1
        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
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
        output_padding: Optional[Tuple],
    ) -> relax.Var:
        conv2d_transpose = self.block_builder.emit(
            relax.op.nn.conv2d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                output_padding=output_padding,
                data_layout="NCHW",
                kernel_layout="IOHW",
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
        output_padding = args[5] if len(args) > 5 else 0
        groups = args[6] if len(args) > 6 else 1
        dilation = args[7] if len(args) > 7 else 1
        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
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

    def _cross_entropy_loss(
        self,
        preds: relax.Expr,
        targets: relax.Expr,
        weights: Optional[relax.Expr],
        reduction: str,
        ignore_index: int,
    ) -> relax.Expr:
        log_probs = relax.op.nn.log_softmax(preds)
        return self.block_builder.emit(
            relax.op.nn.nll_loss(
                log_probs,
                targets,
                weights,
                reduction,
                ignore_index,
            )
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
        import numpy as np  # type: ignore
        from torch.fx.immutable_collections import immutable_list

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

    def _max_pool1d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int]] = 1,
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        # Expand to 3D by adding batch dim if input is 2D
        x_ndim = x.struct_info.ndim
        if x_ndim == 2:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
            relax.op.nn.max_pool1d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCW",
            )
        )

        # Remove added batch dim from result
        if x_ndim == 2:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool1d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool2d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        # Expand to 4D by adding batch dim if input is 3D
        x_ndim = x.struct_info.ndim
        if x_ndim == 3:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
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

        # Remove added batch dim from result
        if x_ndim == 3:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _max_pool3d_impl(
        self,
        x: relax.Expr,
        kernel_size: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        # Expand to 5D by adding batch dim if input is 4D
        x_ndim = x.struct_info.ndim
        if x_ndim == 4:
            x = relax.op.expand_dims(x, axis=0)

        stride = kernel_size if stride is None else stride

        result = self.block_builder.emit(
            relax.op.nn.max_pool3d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCDHW",
            )
        )

        # Remove added batch dim from result
        if x_ndim == 4:
            result = relax.op.squeeze(result, axis=[0])
        return result

    def _max_pool3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False
        return self._max_pool3d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _pad(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        pad = node.args[1]
        mode = node.args[2] if len(node.args) > 2 else node.kwargs.get("mode", "constant")
        value = node.args[3] if len(node.args) > 3 else node.kwargs.get("value", 0.0)
        value = 0.0 if value is None else value

        # Calculate symmetric padding width for each dimension
        # and applying them in reverse order to match the input dimensions.
        input_ndim = x.struct_info.ndim
        pad_width = [0] * (input_ndim * 2)
        pad_pairs = [pad[i : i + 2] for i in range(0, len(pad), 2)]
        reversed_pairs = list(reversed(pad_pairs))
        flattened = [value for pair in reversed_pairs for value in pair]
        pad_width[-len(flattened) :] = flattened

        return self.block_builder.emit(relax.op.nn.pad(x, pad_width, mode, value))

    def _pixel_shuffle(self, node: fx.Node) -> relax.Var:
        data = self.env[node.args[0]]
        upscale_factor = node.args[1]
        assert isinstance(
            upscale_factor, int
        ), "PixelShuffle only accepts an integer upscale_factor."

        return self.block_builder.emit(relax.op.nn.pixel_shuffle(data, upscale_factor))

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

    def _unbind(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        assert isinstance(dim, int), "Expected 2nd argument of unbind as int"
        selections = self.shape_of(x)[dim].value
        ret, split = [], self.block_builder.emit(relax.op.split(x, selections, dim))
        for i in range(selections):
            ret.append(self.block_builder.emit(relax.op.squeeze(split[i], axis=dim)))
        return self.block_builder.emit(relax.Tuple(ret))

    ########## Statistical ##########

    def _mean(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.mean(x, dim, keepdims=keepdim))

    def _norm(self, node: fx.Node) -> relax.Var:
        data = self.env[node.args[0]]
        dtype = data.struct_info.dtype
        order = node.args[1] if len(node.args) > 1 else node.kwargs.get("p", 2)
        axis = node.args[2] if len(node.args) > 2 else None
        keepdims = node.args[3] if len(node.args) > 3 else False

        if order == float("inf"):
            return self.block_builder.emit(
                relax.op.max(relax.op.abs(data), axis=axis, keepdims=keepdims)
            )
        elif order == float("-inf"):
            return self.block_builder.emit(
                relax.op.min(relax.op.abs(data), axis=axis, keepdims=keepdims)
            )
        # frobenius_norm
        elif order == "fro":
            return self.block_builder.emit(
                relax.op.sqrt(
                    relax.op.sum(relax.op.multiply(data, data), axis=axis, keepdims=keepdims)
                )
            )
        else:
            ord_expr = (
                order if isinstance(order, relax.Expr) else relax.const(float(order), dtype=dtype)
            )
            reci_order = (
                relax.op.divide(relax.const(1.0, dtype), ord_expr)
                if isinstance(order, relax.Expr)
                else relax.const(1.0 / order, dtype=dtype)
            )
            return self.block_builder.emit(
                relax.op.power(
                    relax.op.sum(
                        relax.op.power(relax.op.abs(data), ord_expr), axis=axis, keepdims=keepdims
                    ),
                    reci_order,
                )
            )

    def _prod(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.prod(x, dim, keepdims=keepdim))

    def _std(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.std(x, dim, keepdims=keepdim))

    def _sum(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        keepdim = node.kwargs["keepdim"] if "keepdim" in node.kwargs else False
        if len(args) == 1:
            return self.block_builder.emit(relax.op.sum(args[0], keepdims=keepdim))
        return self.block_builder.emit(relax.op.sum(args[0], args[1]))

    def _var(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.variance(x, dim, keepdims=keepdim))

    ########## Search ##########

    def _argmax_argmin(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node):
            x = self.env[node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
            return self.block_builder.emit(op(x, dim, keepdim))

        return convert

    def _where(self, node: fx.Node) -> relax.Var:
        condition = self.env[node.args[0]]
        x = self.env[node.args[1]]
        y = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.where(condition, x, y))

    ########## Manipulation ##########

    def _argsort(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        descending = node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
        return self.block_builder.emit(relax.op.argsort(x, dim, descending))

    def _broadcast_to(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        shape = args[1] if len(args) > 1 else args[0]
        return self.block_builder.emit(relax.op.broadcast_to(x, shape))

    def _cat(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(args[0], axis=axis))

    def _chunk(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        chunks = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        x_shape = self.shape_of(x)
        max_chunks = x_shape[dim].value
        n_sections = min(chunks, max_chunks)
        return self.block_builder.emit(
            relax.op.split(x=x, indices_or_sections=n_sections, axis=dim)
        )

    def _cumprod(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None

        return self.block_builder.emit(relax.op.cumprod(x, dim, dtype))

    def _cumsum(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None
        if "out" in node.kwargs:
            raise ValueError("specifying out for cumsum is not supported yet")

        return self.block_builder.emit(relax.op.cumsum(x, dim, dtype))

    def _expand(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        sizes = args[1:] if len(args) > 2 else args[1]
        broadcast_shape, in_shape = [], self.shape_of(args[0])
        for idx, i in enumerate(sizes):
            if isinstance(i, int) and i == -1:
                broadcast_shape.append(in_shape[idx])
            else:
                broadcast_shape.append(i)
        return self.block_builder.emit(relax.op.broadcast_to(args[0], broadcast_shape))

    def _expand_as(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        # args[0] is the 'self' tensor
        # args[1] is the 'other' tensor
        data = args[0]
        other_shape = self.shape_of(args[1])  # the shape of 'other'
        return self.block_builder.emit(relax.op.broadcast_to(data, other_shape))

    def _flatten_impl(self, x, start_dim, end_dim) -> relax.Var:
        shape = self.shape_of(x)
        start_dim = start_dim if start_dim >= 0 else len(shape) + start_dim
        end_dim = end_dim if end_dim >= 0 else len(shape) + end_dim
        flattened = reduce(lambda x, y: x * y, [shape[i] for i in range(start_dim, end_dim + 1)])
        new_shape = (
            [shape[i] for i in range(0, start_dim)]
            + [flattened]
            + [shape[i] for i in range(end_dim + 1, len(shape))]
        )
        return self.block_builder.emit(relax.op.reshape(x, new_shape))

    def _flatten(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        start_dim = node.args[1] if len(node.args) >= 2 else node.kwargs.get("start_dim", 0)
        end_dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("end_dim", -1)
        return self._flatten_impl(x, start_dim, end_dim)

    def _flip(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims", None)
        if isinstance(dims, (list, tuple)) and len(dims) > 0:
            dims = dims[0]
        elif not isinstance(dims, int):
            raise TypeError(f"flip expects an integer axis, but got {type(dims)}: {dims}")
        return self.block_builder.emit(relax.op.flip(x, dims))

    def _gather(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        index = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.gather_elements(x, index, axis=dim))

    def _index_put(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        tensor = args[0]
        indices = args[1] if len(args) > 1 else node.kwargs.get("indices")
        values = args[2] if len(args) > 2 else node.kwargs.get("values")
        accumulate = args[3] if len(args) > 3 else node.kwargs.get("accumulate", False)

        if indices is None or values is None:
            raise ValueError("'indices and values' arguments are required for index_put operation")

        if not isinstance(accumulate, bool):
            raise TypeError("'accumulate' must be a boolean value, got {}".format(type(accumulate)))

        if isinstance(indices, (list, tuple)):
            indices = relax.Tuple(indices)
        return self.block_builder.emit(relax.op.index_put(tensor, indices, values, accumulate))

    def _index_tensor(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        indices = args[1]
        return self.block_builder.emit(relax.op.index_tensor(args[0], indices))

    def _meshgrid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        indexing = args[1] if len(node.args) > 1 else node.kwargs.get("indexing", "ij")
        input_list = args[0]

        # Single input: return as-is, meshgrid not applicable.
        if len(input_list) == 1:
            return input_list
        new_inputs = []
        for i, item in enumerate(input_list):
            if item.struct_info.ndim == 1:
                new_inputs.append(item)
            elif item.struct_info.ndim == 0:  # Change scalar value into 1D
                const_tensor = relax.op.reshape(item, (1,))
                new_inputs.append(const_tensor)
            else:
                raise TypeError(f"Unsupported meshgrid input type at index {i}: {type(item)}")

        return self.block_builder.emit(relax.op.meshgrid(new_inputs, indexing=indexing))

    def _slice_scatter(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        src = args[1]
        dim = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
        start = args[3] if len(args) > 3 else node.kwargs.get("start", 0)
        end = args[4] if len(args) > 4 else node.kwargs.get("end", self.shape_of(input_tensor)[dim])
        step = args[5] if len(args) > 5 else node.kwargs.get("step", 1)

        return self.block_builder.emit(
            relax.op.slice_scatter(input_tensor, src, start, end, step, axis=dim)
        )

    def _permute(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.permute_dims(x, dims))

    def _repeat(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _roll(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        shifts = args[1] if len(node.args) > 1 else node.kwargs.get("shifts", None)
        dims = args[2] if len(node.args) > 2 else node.kwargs.get("dims", None)

        # Get original shape
        original_shape = self.shape_of(input_tensor)

        def to_int(val):
            if isinstance(val, tir.IntImm):
                return int(val.value)
            elif isinstance(val, int):
                return val
            elif hasattr(val, "__int__"):
                return int(val)
            raise TypeError(f"Unsupported type for shift/dim: {type(val)}")

        def roll_single_dim(tensor: relax.Var, shift: int, dim: int) -> relax.Var:
            shape = self.shape_of(tensor)

            dim_size = shape.values[dim]
            shift_val = to_int(shift)
            dim_size_val = to_int(dim_size)
            shift_mod = shift_val % dim_size_val
            if shift_mod == 0:
                return tensor

            split_pos = dim_size_val - shift_mod
            part1 = self.block_builder.emit(
                relax.op.strided_slice(
                    tensor,
                    axes=[dim],
                    begin=[0],
                    end=[split_pos],
                    strides=[1],
                )
            )
            part2 = self.block_builder.emit(
                relax.op.strided_slice(
                    tensor,
                    axes=[dim],
                    begin=[split_pos],
                    end=[dim_size_val],
                    strides=[1],
                )
            )
            return self.block_builder.emit(relax.op.concat([part2, part1], axis=dim))

        # Handle dims=None (flatten -> roll -> reshape)
        if dims is None:
            flattened = self.block_builder.emit(relax.op.reshape(input_tensor, (-1,)))
            shift_scalar = to_int(shifts[0] if isinstance(shifts, (list, tuple)) else shifts)
            rolled = roll_single_dim(flattened, shift_scalar, 0)
            return self.block_builder.emit(relax.op.reshape(rolled, original_shape))

        # Normalize shifts and dims
        if isinstance(shifts, (list, tuple)):
            shifts = [to_int(s) for s in shifts]
        else:
            shifts = [to_int(shifts)]

        if isinstance(dims, (list, tuple)):
            dims = [to_int(d) for d in dims]
        else:
            dims = [to_int(dims)]

        if len(shifts) != len(dims):
            raise ValueError("shifts and dims must have the same length")

        result = input_tensor
        rank = len(original_shape.values)
        for shift, dim in zip(shifts, dims):
            if dim < 0:
                dim += rank
            result = roll_single_dim(result, shift, dim)

        return result

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.reshape(x, dims))

    def _reshape_as(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        other = args[1]
        dims = self.shape_of(other)
        return self.block_builder.emit(relax.op.reshape(x, dims))

    def _scatter(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if len(node.args) == 1:
            dim = node.kwargs["dim"]
            index = self.env[node.kwargs["index"]]
            src = self.env[node.kwargs["src"]]
        elif len(node.args) == 4:
            dim = node.args[1]
            index = self.env[node.args[2]]
            src = self.env[node.args[3]]
        else:
            raise Exception("Unexpected args " + str(node.args))
        return self.block_builder.emit(relax.op.scatter_elements(x, index, src, axis=dim))

    def _sort(self, node: fx.Node) -> relax.Var:
        # torch.sort() returns a tuple of values and indices
        # we use argsort to get indices and gather_elements to get values
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        descending = node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)

        indices = self.block_builder.emit(relax.op.argsort(x, dim, descending))
        values = self.block_builder.emit(relax.op.gather_elements(x, indices, axis=dim))
        return self.block_builder.emit(relax.Tuple([values, indices]))

    def _split(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        split_size = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(split_size, (list, tuple)):
            n_section = []
            for s in split_size[:-1]:
                cum_sum = 0 if not n_section else n_section[-1]
                n_section.append(s + cum_sum)
        else:
            n_section = (self.shape_of(x)[dim].value + split_size - 1) // split_size
        return self.block_builder.emit(relax.op.split(x, n_section, dim))

    def _squeeze(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        return self.block_builder.emit(relax.op.squeeze(x, dim))

    def _stack(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        tensor_list = args[0]
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.stack(tensor_list, axis=axis))

    def _take(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        indices = self.env[node.args[1]]
        indices = self.block_builder.emit(relax.op.astype(indices, "int32"))
        return self.block_builder.emit(relax.op.take(x, indices))

    def _tile(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _topk(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        k = args[1] if len(args) > 1 else node.kwargs.get("k", 1)
        dim = args[2] if len(args) > 2 else node.kwargs.get("dim", -1)
        largest = args[3] if len(args) > 3 else node.kwargs.get("largest", True)
        _sorted = args[4] if len(args) > 4 else node.kwargs.get("_sorted", True)

        if not _sorted:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        return self.block_builder.emit(
            relax.op.topk(x, k=k, axis=dim, largest=largest, ret_type="both", dtype="int64")
        )

    def _transpose(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        full_idx = list(range(len(self.shape_of(args[0]))))
        full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
        return self.block_builder.emit(relax.op.permute_dims(args[0], full_idx))

    ########## Creation ##########

    def _detach(self, node: fx.Node) -> relax.Var:
        # There is no way to implement detach() such that the output shares
        # the same memory as the input. In-place operations are not supported
        # by the translator, and therefore we just return a copy of the input.
        return self.env[node.args[0]]

    def _copy_(self, node: fx.Node) -> relax.Var:
        # Copies the source tensor's into the destination tensor
        # In TVM, that means simply returning the source tensor
        return self.env[node.args[1]]

    def _to_copy(self, node: fx.Node) -> relax.Var:
        # Returns a copy of the input tensor
        import torch  # type: ignore

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = self._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = self._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _arange(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        start_end_step = [None, None, None]
        if "start" in node.kwargs:
            start_end_step[0] = node.kwargs["start"]
        if "end" in node.kwargs:
            start_end_step[1] = node.kwargs["end"]
        if "step" in node.kwargs:
            start_end_step[2] = node.kwargs["step"]

        if len(node.args) == 1:
            assert start_end_step[1] is None
            start_end_step[1] = node.args[0]
        elif len(node.args) == 2:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
        elif len(node.args) == 3:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            assert start_end_step[2] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
            start_end_step[2] = node.args[2]

        if start_end_step[0] is None:
            start_end_step[0] = 0
        if start_end_step[2] is None:
            start_end_step[2] = 1

        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        elif any([isinstance(x, float) for x in start_end_step]):
            dtype = self._convert_data_type(torch.get_default_dtype())
        else:
            dtype = "int64"
        start_end_step = [
            self.env[x] if isinstance(x, torch.fx.Node) else x for x in start_end_step
        ]
        return self.block_builder.emit(relax.op.arange(*start_end_step, dtype=dtype))

    def _empty(self, node: fx.Node) -> relax.Var:
        dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        return self.block_builder.emit(relax.op.zeros(node.args[0], dtype))

    def _empty_like(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.zeros_like(x))

    def _eye(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        n = args[0]
        m = args[1] if len(args) > 1 else n
        dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        return self.block_builder.emit(relax.op.eye(n, m, dtype=dtype))

    def _fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        value = args[1] if isinstance(args[1], relax.Expr) else relax.const(args[1], dtype)
        return self.block_builder.emit(relax.op.full(x.struct_info.shape, value, dtype))

    def _inplace_fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        value = args[1] if isinstance(args[1], relax.Expr) else relax.const(args[1], dtype)
        filled = self.block_builder.emit(relax.op.full(x.struct_info.shape, value, dtype))
        self.env[node.args[0]] = filled
        return filled

    def _full(self, node: fx.Node) -> relax.Var:
        import torch

        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], (list, tuple)) else (args[0],))
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        value = args[1] if isinstance(args[1], relax.expr.Constant) else relax.const(args[1], dtype)
        return self.block_builder.emit(
            relax.op.full(
                size,
                value,
                dtype,
            )
        )

    def _full_like(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        fill_value = relax.const(node.args[1])
        return self.block_builder.emit(relax.op.full_like(x, fill_value))

    def _index_select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _inplace_masked_fill(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        mask = self.env[node.args[1]]
        value = node.args[2]
        rx_value = relax.const(value)
        values = self.block_builder.emit(relax.op.full_like(x, rx_value))
        output = self.block_builder.emit(relax.op.where(mask, values, x))
        self.env[node.args[0]] = output
        return output

    def _linspace(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        start = args[0]
        stop = args[1]
        step = args[2]

        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + (step / 2)
        else:
            stop = start + step

        if len(args) <= 3 or args[3] is None:
            import torch

            dtype = self._convert_data_type(str(torch.get_default_dtype()))
        else:
            dtype = self._convert_data_type(args[3])

        return self.block_builder.emit(
            relax.op.arange(start=start, end=stop, step=step, dtype=dtype)
        )

    def _masked_fill(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        mask = self.env[node.args[1]]
        rx_value = relax.const(node.args[2])
        values = self.block_builder.emit(relax.op.full_like(x, rx_value))
        return self.block_builder.emit(relax.op.where(mask, values, x))

    def _new_ones(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        self_var = args[0]
        size = args[1] if isinstance(args[1], (list, tuple)) else args[1:]
        if not isinstance(size, (list, tuple)):
            size = (size,)
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, self_var.struct_info.dtype),
                self_var.struct_info.dtype,
            )
        )

    def _new_zeros(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        size = (
            args[1]
            if isinstance(args[1], (list, tuple))
            else (args[1],)
            if len(args[1:]) == 1
            else args[1:]
        )
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(0, input_tensor.struct_info.dtype),
                input_tensor.struct_info.dtype,
            )
        )

    def _ones(self, node: fx.Node) -> relax.Var:
        import torch

        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], (list, tuple)) else (args[0],))
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, dtype),
                dtype,
            )
        )

    ########## DataType ##########

    def _to(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = BaseFXGraphImporter._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = BaseFXGraphImporter._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _type_as(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        other = self.env[node.args[1]]
        dtype = other.struct_info.dtype
        return self.block_builder.emit(relax.op.astype(x, dtype))

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

    def _item(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.take(x, relax.const(0, "int64"), axis=0))

    def _zeros_inplace(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output = self.block_builder.emit(relax.op.zeros_like(x))
        self.env[node.args[0]] = output
        return output

    def _zeros_like(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.zeros_like(x))

    @abc.abstractmethod
    def create_convert_map(
        self,
    ) -> Dict[Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]]:
        """Create convert map"""
