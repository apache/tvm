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
"""PyTorch FX frontend of Relax."""
from typing import Callable, Dict, List, Tuple, Union
from functools import partial, reduce

import tvm
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class TorchFXImporter(BaseFXGraphImporter):
    """An importer from PyTorch FX to Relax."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore

        super().__init__()
        self.named_modules: Dict[str, torch.Module] = None

    ########## Utilities ##########

    def _fetch_attr(self, model, target: str):
        import torch  # type: ignore

        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non existing target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            # Its possible for the resulting tensor to be a parameter.
            # If so, return the parameter instead.
            if attr_itr in self.params:
                return self.params[attr_itr]
            return self._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    ########## Unary Ops ##########

    def _leakyrelu_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        alpha = module.negative_slope
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        dim = module.dim
        assert dim is not None
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _softmax_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        dim = module.dim
        assert dim is not None
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _inplace_tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else 0
            assert isinstance(k, int)

            mutated = self.block_builder.emit(op(x, k))
            self.env[node.args[0]] = mutated
            return mutated

        return convert

    ########## Neural Network ##########

    def _adaptive_avg_pool2d_module(self, node: fx.Node) -> relax.Var:

        module = self.named_modules[node.target]
        x = self.env[node.args[0]]
        output_size = module.output_size
        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )

    def _avg_pool2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        ceil_mode = module.ceil_mode
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _batch_norm_2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params[module.bias]
        running_mean = self._convert_torch_tensor_to_relax(module.running_mean)
        running_var = self._convert_torch_tensor_to_relax(module.running_var)
        eps = module.eps

        res_tuple = self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
            )
        )

        return self.block_builder.emit(relax.TupleGetItem(res_tuple, 0))

    def _conv_transpose1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv_transpose2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv1d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _conv3d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)

        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _cross_entropy(self, node: fx.Node) -> relax.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        weights = self.env.get(node.kwargs["weight"], None)
        reduction = node.kwargs["reduction"]
        ignore_index = node.kwargs["ignore_index"]

        return self.block_builder.emit(
            relax.op.nn.nll_loss(
                relax.op.nn.log_softmax(preds), targets, weights, reduction, ignore_index
            )
        )

    def _cross_entropy_module(self, node: fx.Node) -> relax.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        module = self.named_modules[node.target]

        weights = module.weight
        if weights is not None:
            if weights in self.params:
                weights = self.params[weights]
            else:
                weights = relax.const(weights.numpy(), preds.struct_info.dtype)

        reduction = module.reduction
        ignore_index = module.ignore_index

        return self.block_builder.emit(
            relax.op.nn.nll_loss(
                relax.op.nn.log_softmax(preds), targets, weights, reduction, ignore_index
            )
        )

    def _embedding_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        return self._embedding_impl(x, weight)

    def _group_norm_module(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        num_groups = module.num_groups
        if module.affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.num_channels), x.checked_type)
            beta = relax.const(torch.zeros_like(module.num_channels), x.checked_type)
        eps = module.eps

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _interpolate(self, node: fx.Node) -> relax.Var:
        # torch.nn.functional.interpolate(
        #   input, size=None, scale_factor=None, mode='nearest', align_corners=None,
        #   recompute_scale_factor=None, antialias=False)
        # (TODO) this is a temporary implementation for interpolate that only considers NCHW layout
        # it basically replicates the implementation in tvm.relay.frontend.pytorch
        data = self.env[node.args[0]]
        size = (
            node.args[1]
            if len(node.args) > 1
            else (node.kwargs["size"] if "size" in node.kwargs else None)
        )
        scale_factor = (
            node.args[2]
            if len(node.args) > 2
            else (node.kwargs["scale_factor"] if "scale_factor" in node.kwargs else None)
        )
        method = (
            node.args[3]
            if len(node.args) > 3
            else (node.kwargs["mode"] if "mode" in node.kwargs else "nearest")
        )
        align_corners = (
            node.args[4]
            if len(node.args) > 4
            else (node.kwargs["align_corners"] if "align_corners" in node.kwargs else None)
        )
        recompute_scale_factor = (
            node.args[5]
            if len(node.args) > 5
            else (
                node.kwargs["recompute_scale_factor"]
                if "recompute_scale_factor" in node.kwargs
                else None
            )
        )
        antialias = (
            node.args[6]
            if len(node.args) > 6
            else (node.kwargs["antialias"] if "antialias" in node.kwargs else False)
        )

        assert recompute_scale_factor is None
        assert antialias is False

        if size is None:
            shape = self.shape_of(data)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, tuple):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        if method.startswith("nearest"):
            method = "nearest_neighbor"
        elif method[0:2] == "bi":
            method = method[2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return self.block_builder.emit(
            relax.op.image.resize2d(
                data, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _linear_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params.get(module.bias, None)
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool2d_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        ceil_mode = module.ceil_mode

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    ########## Manipulation ##########

    def _chunk(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        chunks = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.split(x, chunks, dim))

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

    def _flatten_module(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        start_dim = module.start_dim
        end_dim = module.end_dim
        return self._flatten_impl(x, start_dim, end_dim)

    def _size(self, node: fx.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        if len(node.args) == 1:
            assert isinstance(shape, relax.ShapeExpr)
            return shape
        assert len(node.args) == 2
        idx = node.args[1]
        return self.shape_of(x)[idx].value

    ########## Creation ##########

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

    def _masked_fill(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        mask = self.env[node.args[1]]
        rx_value = relax.const(node.args[2])
        values = self.block_builder.emit(relax.op.full_like(x, rx_value))
        return self.block_builder.emit(relax.op.where(mask, values, x))

    def _masked_scatter(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        mask = self.env[node.args[1]]
        source = self.env[node.args[2]]
        ndim = len(mask.struct_info.shape)
        if ndim == 1:
            index = self.block_builder.emit(relax.op.cumsum(mask, 0, dtype="int32"))
            index = self.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            gathered_source = self.block_builder.emit(relax.op.take(source, index, axis=0))
        else:
            f_mask = self.block_builder.emit(relax.op.reshape(mask, [-1]))
            index = self.block_builder.emit(relax.op.cumsum(f_mask, 0, dtype="int32"))
            index = self.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            source_shape = [-1] + [
                s for idx, s in enumerate(source.struct_info.shape) if idx >= ndim
            ]
            f_source = self.block_builder.emit(relax.op.reshape(source, source_shape))
            gathered_source = self.block_builder.emit(relax.op.take(f_source, index, axis=0))
            gathered_source = self.block_builder.emit(
                relax.op.reshape(gathered_source, x.struct_info.shape)
            )
        if ndim != len(x.struct_info.shape):
            mask = self.block_builder.emit(relax.op.broadcast_to(mask, x.struct_info.shape))
        return self.block_builder.emit(relax.op.where(mask, gathered_source, x))

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

    def _tensor(self, node: fx.Node) -> relax.Var:
        dtype = node.kwargs.get("dtype", None)
        if isinstance(node.args[0], float):
            return relax.const(node.args[0], dtype if dtype is not None else "float32")
        elif isinstance(node.args[0], int):
            return relax.const(node.args[0], dtype if dtype is not None else "int64")
        raise ValueError("torch.tensor with value not a float or int is not accepted")

    ########## DataType ##########

    def _float(self, node: fx.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float32"))

    def _half(self, node: fx.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float16"))

    def _to(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = TorchFXImporter._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = TorchFXImporter._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _type(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = TorchFXImporter._convert_data_type(node.args[1], self.env)
        return self.block_builder.emit(relax.op.astype(x, dtype))

    ########## Others ##########

    def _getattr(self, node: fx.Node) -> relax.Var:
        if isinstance(self.env[node.args[0]], relax.Expr):
            if node.args[1] == "dtype":
                return self.env[node.args[0]].struct_info.dtype
            elif node.args[1] == "shape":
                return self.shape_of(self.env[node.args[0]])
        return getattr(self.env[node.args[0]], node.args[1])

    def _sym_size_int(self, node: fx.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        idx = node.args[1]
        return self.block_builder.emit(relax.const(shape[idx].value, "int32"))

    def create_input_vars(self, input_info: List[Tuple[Tuple[int], str]]) -> List[relax.Var]:
        inputs = list()
        for idx, (shape, dtype) in enumerate(input_info):
            inputs.append(
                relax.Var(
                    f"inp_{idx}", relax.TensorStructInfo(shape, self._convert_data_type(dtype))
                )
            )
        return inputs

    def create_convert_map(
        self,
    ) -> Dict[Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]]:
        import operator
        from torch import nn

        return {
            ## call_module
            # unary
            nn.Dropout: lambda node: self.env[node.args[0]],
            nn.GELU: self._gelu,
            nn.Hardsigmoid: self._hardsigmoid,
            nn.Hardswish: self._hardswish,
            nn.Identity: lambda node: self.env[node.args[0]],
            nn.LeakyReLU: self._leakyrelu_module,
            nn.LogSoftmax: self._log_softmax_module,
            nn.ReLU: self._unary_op(relax.op.nn.relu),
            nn.ReLU6: lambda node: self.block_builder.emit(
                relax.op.clip(self.env[node.args[0]], 0, 6)
            ),
            nn.Sigmoid: self._unary_op(relax.op.sigmoid),
            nn.SiLU: self._unary_op(relax.op.nn.silu),
            nn.Softmax: self._softmax_module,
            nn.Tanh: self._unary_op(relax.op.tanh),
            # neural network
            nn.AdaptiveAvgPool2d: self._adaptive_avg_pool2d_module,
            nn.AvgPool2d: self._avg_pool2d_module,
            nn.BatchNorm2d: self._batch_norm_2d_module,
            nn.Conv1d: self._conv1d_module,
            nn.Conv2d: self._conv2d_module,
            nn.Conv3d: self._conv3d_module,
            nn.ConvTranspose1d: self._conv_transpose1d_module,
            nn.ConvTranspose2d: self._conv_transpose2d_module,
            nn.CrossEntropyLoss: self._cross_entropy_module,
            nn.GroupNorm: self._group_norm_module,
            nn.LayerNorm: self._layer_norm_module,
            nn.Linear: self._linear_module,
            nn.MaxPool2d: self._max_pool2d_module,
            nn.modules.sparse.Embedding: self._embedding_module,
            # tensor manipulation
            nn.Flatten: self._flatten_module,
            ## call_function and call_method
            # unary
            "acos": self._unary_op(relax.op.acos),
            "acosh": self._unary_op(relax.op.acosh),
            "asin": self._unary_op(relax.op.asin),
            "asinh": self._unary_op(relax.op.asinh),
            "atan": self._unary_op(relax.op.atan),
            "atanh": self._unary_op(relax.op.atanh),
            "clamp": self._clamp,
            "cos": self._unary_op(relax.op.cos),
            "cosh": self._unary_op(relax.op.cosh),
            "dropout": lambda node: self.env[node.args[0]],
            "exp": self._unary_op(relax.op.exp),
            "gelu": self._gelu,
            "hardsigmoid": self._hardsigmoid,
            "hardswish": self._hardswish,
            "leaky_relu": self._leakyrelu,
            "log_softmax": self._log_softmax,
            "neg": self._unary_op(relax.op.negative),
            "relu": self._unary_op(relax.op.nn.relu),
            "round": self._round,
            "rsqrt": self._unary_op(relax.op.rsqrt),
            "sigmoid": self._unary_op(relax.op.sigmoid),
            "silu": self._unary_op(relax.op.nn.silu),
            "sin": self._unary_op(relax.op.sin),
            "sinh": self._unary_op(relax.op.sinh),
            "softmax": self._softmax,
            "sqrt": self._unary_op(relax.op.sqrt),
            "tan": self._unary_op(relax.op.tan),
            "tanh": self._unary_op(relax.op.tanh),
            "tril_": self._inplace_tril_triu(relax.op.tril),
            "tril": self._tril_triu(relax.op.tril),
            "triu_": self._inplace_tril_triu(relax.op.triu),
            "triu": self._tril_triu(relax.op.triu),
            # binary
            "add": self._binary_op(relax.op.add, operator.add),
            "eq": self._binary_op(relax.op.equal, operator.eq),
            "floordiv": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "iadd": self._binary_op(relax.op.add, operator.add),
            "lt": self._binary_op(relax.op.less, operator.lt),
            "matmul": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max": self._binary_op(relax.op.maximum, max),
            "mul": self._binary_op(relax.op.multiply, operator.mul),
            "pow": self._binary_op(relax.op.power, operator.pow),
            "sub": self._binary_op(relax.op.subtract, operator.sub),
            "truediv": self._binary_op(relax.op.divide, operator.truediv),
            # neural network
            "adaptive_avg_pool2d": self._adaptive_avg_pool2d,
            "addmm": self._addmm,
            "avg_pool2d": self._avg_pool2d,
            "baddbmm": self._baddbmm,
            "bmm": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "conv_transpose1d": self._conv_transpose1d,
            "conv_transpose2d": self._conv_transpose2d,
            "conv1d": self._conv1d,
            "conv2d": self._conv2d,
            "conv3d": self._conv3d,
            "cross_entropy": self._cross_entropy,
            "einsum": self._einsum,
            "interpolate": self._interpolate,
            "layer_norm": self._layer_norm,
            "linear": self._linear,
            "max_pool2d": self._max_pool2d,
            "scaled_dot_product_attention": self._scaled_dot_product_attention,
            "stochastic_depth": lambda node: self.env[node.args[0]],
            "unbind": self._unbind,
            # statistical
            "mean": self._mean,
            "sum": self._sum,
            # search
            "argmax": self._argmax_argmin(relax.op.argmax),
            "argmin": self._argmax_argmin(relax.op.argmin),
            # tensor manipulation
            "cat": self._cat,
            "chunk": self._chunk,
            "concat": self._cat,
            "contiguous": lambda node: self.env[node.args[0]],
            "cumsum": self._cumsum,
            "expand": self._expand,
            "flatten": self._flatten,
            "permute": self._permute,
            "repeat": self._repeat,
            "reshape": self._reshape,
            "scatter": self._scatter,
            "size": self._size,
            "split": self._split,
            "squeeze": self._squeeze,
            "stack": self._stack,
            "tile": self._tile,
            "transpose": self._transpose,
            "unsqueeze": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view": self._reshape,
            # tensor creation
            "arange": self._arange,
            "empty": self._empty,
            "fill_": self._inplace_fill,
            "full": self._full,
            "index_select": self._index_select,
            "masked_fill_": self._inplace_masked_fill,
            "masked_fill": self._masked_fill,
            "masked_scatter": self._masked_scatter,
            "new_ones": self._new_ones,
            "ones": self._ones,
            "tensor": self._tensor,
            # datatype
            "astype": self._type,
            "float": self._float,
            "half": self._half,
            "to": self._to,
            "type": self._type,
            # other
            "getattr": self._getattr,
            "getitem": self._getitem,
            "sym_size.int": self._sym_size_int,
        }

    def update_convert_map(self, custom_convert_map: dict):
        """Update self.convert_map with custom convert map

        Parameters
        ----------
        custom_convert_map : Dictionary of str to Relax op
            A custom op conversion map in the same format as self.convert_map
        """

        self.convert_map.update(custom_convert_map)

    def from_fx(
        self,
        model,
        input_info: List[Tuple[Tuple[int], str]],
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
        custom_convert_map: dict = None,
    ) -> tvm.IRModule:
        """Convert a PyTorch FX GraphModule to a Relax program."""
        from torch import fx

        if custom_convert_map:
            custom_ops = set(custom_convert_map.keys())
            self.update_convert_map(custom_convert_map)
        else:
            custom_ops = set()
        self.named_modules = dict(model.named_modules())

        graph: fx.Graph = model.graph

        # Create input variables.
        inputs = self.create_input_vars(input_info)

        # Initialize the block builder with a function and a dataflow block.
        func_name = "main"
        self.block_builder = relax.BlockBuilder()
        params = []
        if keep_params_as_input:
            func_attrs = {"num_input": len(inputs)}
            for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                shape = param.data.shape
                dtype = self._convert_data_type(str(param.data.dtype))
                inputs.append(relax.Var(name, relax.TensorStructInfo(shape, dtype)))
                self.params[param] = inputs[-1]
                params.append(tvm.nd.array(param.data.cpu().numpy()))
        else:
            func_attrs = None

        with self.block_builder.function(name=func_name, params=inputs.copy(), attrs=func_attrs):
            output = None
            with self.block_builder.dataflow():
                # Translate model parameters.
                for _, param in model.named_parameters():
                    shape = param.data.shape
                    dtype = self._convert_data_type(str(param.data.dtype))
                    if dtype in ("float32", "float16"):
                        if not keep_params_as_input:
                            self.params[param] = self._convert_torch_tensor_to_relax(param)
                    else:
                        raise ValueError("Unsupported data type for model parameters: %s" % dtype)
                # Translate the model.
                for node in graph.nodes:
                    if node.op == "placeholder":
                        assert len(inputs) > 0, "Provided inputs is less than actual inputs"
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Ignore sym input
                            continue

                        self.env[node] = inputs.pop(0)
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1

                        # return tuple
                        if isinstance(args[0], (tuple, list, relax.Tuple)):
                            # unit tuple
                            if unwrap_unit_return_tuple and len(args[0]) == 1:
                                output = self.block_builder.emit_output(args[0][0])
                            elif no_bind_return_tuple:
                                output = []
                                for ret in args[0]:
                                    output.append(self.block_builder.emit_output(ret))

                        if output is None:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = self._fetch_attr(model, node.target)
                    elif node.op == "call_module":
                        module = self.named_modules[node.target]
                        assert (
                            type(module) in self.convert_map
                        ), f"Unsupported module type {type(module)}"
                        self.env[node] = self.convert_map[type(module)](node)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        assert (
                            func_name in self.convert_map
                        ), f"Unsupported function type {func_name}"
                        if func_name in custom_ops:
                            self.env[node] = self.convert_map[func_name](node, self)
                        else:
                            self.env[node] = self.convert_map[func_name](node)
                    elif node.op == "call_method":
                        assert (
                            node.target in self.convert_map
                        ), f"Unsupported function target {node.target}"
                        self.env[node] = self.convert_map[node.target](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        if keep_params_as_input:
            mod["main"] = mod["main"].with_attr("params", params)
        return mod


def from_fx(
    model,
    input_info: List[Tuple[Tuple[int], str]],
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
    custom_convert_map: dict = None,
) -> tvm.IRModule:
    """Convert a PyTorch FX GraphModule to a Relax program

    Parameters
    ----------
    model : fx.GraphModule
        The PyTorch FX GraphModule to convert.

    input_info : List[Tuple[Tuple[int], str]]
        A list of shapes and data types of input tensors.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    custom_convert_map : Dictionary of str to Relax op
        A custom op conversion map in the same format as TorchFXImporter.convert_map

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.
        If `keep_params_as_input` is true, the "main" function have an attribute
        "params" that contains the weights of the input model. The weights
        can be detached by `relax.frontend.detach_params`.

    Examples
    --------
    Users can use the FX tracer or dynamo.export() to extract
    a fx.GraphModule from a PyTorch model. The following codes show
    how to convert a PyTorch model to a Relax program.

    .. code-block:: python

        # Import the importer.
        import numpy as np
        import torch
        from tvm.relax.frontend.torch_fx import from_fx
        from torch import _dynamo as dynamo

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()
        input_info = [((128, 10), "float32")]
        input_tensors = [
            torch.astensor(np.random.randn(*shape).astype(dtype))
            for shape, dtype in input_info
        ]

        # Use FX tracer to trace the PyTorch model.
        graph_module = fx.symbolic_trace(torch_model)

        # Use the dynamo.export() to export the PyTorch model to FX.
        try:
            graph_module = dynamo.export(torch_model, *input_tensors)
        except:
            raise RuntimeError("Failed to export the PyTorch model to FX.")

        # Use the importer to import the PyTorch model to Relax.
        mod: tvm.IRModule = from_fx(graph_module, input_info)

        # Print out the imported model.
        print(mod.script())

    Notes
    -----
    For a given PyTorch model, to lookup the names of the model inputs in
    FX, one can use

    .. code-block:: python

        fx.symbolic_trace(model).graph.print_tabular()

    to print out the tabular representation of the PyTorch module, and then
    check the placeholder rows in the beginning of the tabular.
    """
    return TorchFXImporter().from_fx(
        model,
        input_info,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
        custom_convert_map=custom_convert_map,
    )
