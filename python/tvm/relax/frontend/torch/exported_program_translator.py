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
"""PyTorch ExportedProgram of Relax."""
from collections import ChainMap, OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import tvm
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to Relax."""

    from torch import fx

    ########## Unary Ops ##########

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs.get("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs.get("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    def _log2(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relax.op.divide(relax.op.log(x), relax.const(0.6931471805599453, x.struct_info.dtype))
        )

    def _log10(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relax.op.divide(relax.op.log(x), relax.const(2.302585092994046, x.struct_info.dtype))
        )

    def _log1p(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        one = relax.const(1, x.struct_info.dtype)
        return self.block_builder.emit(relax.op.log(relax.op.add(x, one)))

    def _reciprocal(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.divide(relax.const(1.0, x.struct_info.dtype), x))

    ########## Neural Network ##########

    def _batch_norm(self, node: fx.Node, training: bool) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))
        ignore_running_stats = (
            node.args[5] if len(node.args) > 5 else node.kwargs.get("track_running_stats", True)
        )
        track_running_stats = not ignore_running_stats
        momentum = node.args[6] if len(node.args) > 6 else node.kwargs.get("momentum", 0.1)
        eps = node.args[7] if len(node.args) > 7 else node.kwargs.get("eps", 1e-05)

        if track_running_stats:
            training = True

        return self.block_builder.emit(
            relax.op.nn.batch_norm(
                data=x,
                gamma=weight,
                beta=bias,
                moving_mean=running_mean,
                moving_var=running_var,
                axis=1,  # Always over channel
                epsilon=eps,
                momentum=momentum,
                training=training,
            )[0]
        )

    def _batch_norm_legit_functional(self, node: fx.Node) -> relax.Var:
        # This method is called for batch_norm in training mode
        # TODO does not have correctness!
        # TODO we need to store the running mean and variance returned by the
        # previous call to batch_norm and pass it again
        training = True
        return self._batch_norm(node, training)

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        # This method is called for batch_norm in eval mode
        training = False
        return self._batch_norm(node, training)

    def _cross_entropy_default(self, node: fx.Node) -> relax.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        weight = self.env.get(node.args[2], None) if len(node.args) > 2 else None
        reduction = node.kwargs.get("reduction", "mean")
        ignore_index = node.kwargs.get("ignore_index", -100)
        return self._cross_entropy_loss(preds, targets, weight, reduction, ignore_index)

    def _group_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

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

    def _upsample_impl(
        self,
        x: relax.Expr,
        size,
        scale_factor,
        method: str,
        align_corners: bool,
    ) -> relax.Var:
        coord_trans = "align_corners" if align_corners else "half_pixel"

        if size is None:
            shape = self.shape_of(x)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, (tuple, list)):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        return self.block_builder.emit(
            relax.op.image.resize2d(
                x, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _upsample_bilinear2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", 1)
        return self._upsample_impl(
            x, size=size, scale_factor=scale_factor, method="linear", align_corners=align_corners
        )

    def _upsample_nearest2d(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)

        if size:
            scale_factor = None  # Can only define size or scale_factor, not both
            align_corners = (
                node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", None)
            )

        else:
            # TODO figure out why pytorch export passes a list such as
            # [scale_factor,scale_factor] instead of just an int for
            # scale_factor. Using first element for now
            scale_factor = (
                node.args[2][0] if len(node.args) > 2 else node.kwargs.get("scale_factor", 1)
            )
            align_corners = (
                node.args[3] if len(node.args) > 3 else node.kwargs.get("align_corners", None)
            )

        return self._upsample_impl(
            x,
            size=size,
            scale_factor=scale_factor,
            method="nearest_neighbor",
            align_corners=align_corners,
        )

    def _upsample_bicubic2d(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", None)
        )
        if size is not None:
            scale_factor = None
        else:
            scale_arg = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", 1)
            if isinstance(scale_arg, (list, tuple)):
                scale_factor = scale_arg[0]
            else:
                scale_factor = scale_arg

        return self._upsample_impl(
            x,
            size=size,
            scale_factor=scale_factor,
            method="cubic",
            align_corners=align_corners,
        )

    ########## Manipulation ##########

    def _narrow(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        start = node.args[2]
        length = node.args[3]
        return self.block_builder.emit(relax.op.strided_slice(x, [dim], [start], [length]))

    def _select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relax.const(node.args[2], "int64")
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _slice(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

    def _unflatten(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = node.args[1]
        sizes = node.args[2]

        x_shape = list(self.shape_of(x))
        if dim < 0:
            dim += len(x_shape)

        new_shape = x_shape[:dim] + sizes + x_shape[dim + 1 :]
        return self.block_builder.emit(relax.op.reshape(x, new_shape))

    ########## Creation ##########

    def _one_hot(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_classes = node.args[1] if len(node.args) > 1 else node.kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes not found in node.args or node.kwargs")

        on_value = node.args[2] if len(node.args) > 2 else node.kwargs.get("on_value", 1)
        off_value = node.args[3] if len(node.args) > 3 else node.kwargs.get("off_value", 0)
        axis = node.args[4] if len(node.args) > 4 else node.kwargs.get("axis", -1)

        on_value = relax.PrimValue(on_value)
        off_value = relax.PrimValue(off_value)

        return self.block_builder.emit(relax.op.one_hot(x, on_value, off_value, num_classes, axis))

    def _zeros(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], (list, tuple)) else (args[0],))
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(relax.op.zeros(size, dtype))

    ########## Others ##########

    def create_convert_map(
        self,
    ) -> Dict[str, Callable[[fx.Node], relax.Var]]:
        import operator

        return {
            # unary
            "abs.default": self._unary_op(relax.op.abs),
            "acos.default": self._unary_op(relax.op.acos),
            "acosh.default": self._unary_op(relax.op.acosh),
            "asin.default": self._unary_op(relax.op.asin),
            "asinh.default": self._unary_op(relax.op.asinh),
            "atan.default": self._unary_op(relax.op.atan),
            "atanh.default": self._unary_op(relax.op.atanh),
            "bitwise_not.default": self._unary_op(relax.op.bitwise_not),
            "ceil.default": self._unary_op(relax.op.ceil),
            "celu.default": self._celu,
            "clamp.default": self._clamp,
            "clamp_min.default": self._clamp_min,
            "clamp_max.default": self._clamp_max,
            "cos.default": self._unary_op(relax.op.cos),
            "cosh.default": self._unary_op(relax.op.cosh),
            "dropout.default": lambda node: self.env[node.args[0]],
            "dropout_.default": lambda node: self.env[node.args[0]],
            "elu.default": self._elu,
            "erf.default": self._unary_op(relax.op.erf),
            "exp.default": self._unary_op(relax.op.exp),
            "floor.default": self._unary_op(relax.op.floor),
            "gelu.default": self._gelu,
            "hardsigmoid.default": self._hardsigmoid,
            "hardswish.default": self._hardswish,
            "hardswish_.default": self._hardswish,
            "hardtanh.default": self._hardtanh,
            "hardtanh_.default": self._hardtanh,
            "isfinite.default": self._unary_op(relax.op.isfinite),
            "isinf.default": self._unary_op(relax.op.isinf),
            "isin.Tensor_Tensor": self._isin,
            "isnan.default": self._unary_op(relax.op.isnan),
            "leaky_relu.default": self._leakyrelu,
            "leaky_relu_.default": self._leakyrelu,
            "log.default": self._unary_op(relax.op.log),
            "log2.default": self._log2,
            "log10.default": self._log10,
            "log1p.default": self._log1p,
            "logical_not.default": self._unary_op(relax.op.logical_not),
            "log_softmax.int": self._log_softmax,
            "neg.default": self._unary_op(relax.op.negative),
            "pad.default": self._pad,
            "pixel_shuffle.default": self._pixel_shuffle,
            "prelu.default": self._prelu,
            "reciprocal.default": self._reciprocal,
            "relu.default": self._unary_op(relax.op.nn.relu),
            "relu_.default": self._unary_op(relax.op.nn.relu),
            "relu6.default": self._unary_op(relax.op.nn.relu6),
            "relu6_.default": self._unary_op(relax.op.nn.relu6),
            "round.default": self._round,
            "rsqrt.default": self._unary_op(relax.op.rsqrt),
            "rsub.Tensor": self._rsub,
            "rsub.Scalar": self._rsub,
            "selu.default": self._unary_op(relax.op.nn.selu),
            "sigmoid.default": self._unary_op(relax.op.sigmoid),
            "sign.default": self._unary_op(relax.op.sign),
            "silu.default": self._unary_op(relax.op.nn.silu),
            "silu_.default": self._unary_op(relax.op.nn.silu),
            "sin.default": self._unary_op(relax.op.sin),
            "sinh.default": self._unary_op(relax.op.sinh),
            "softmax.int": self._softmax,
            "softplus.default": self._softplus,
            "softshrink.default": self._softshrink,
            "sqrt.default": self._unary_op(relax.op.sqrt),
            "square.default": self._unary_op(relax.op.square),
            "tan.default": self._unary_op(relax.op.tan),
            "tanh.default": self._unary_op(relax.op.tanh),
            "tril.default": self._tril_triu(relax.op.tril),
            "triu.default": self._tril_triu(relax.op.triu),
            "trunc.default": self._unary_op(relax.op.trunc),
            # binary
            "add.Tensor": self._binary_op(relax.op.add, operator.add),
            "add_.Tensor": self._binary_op(relax.op.add, operator.add),
            "bitwise_or_.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_or.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_or_.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_or.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "div.Tensor": self._binary_op(relax.op.divide, operator.truediv),
            "div.Tensor_mode": self._div,
            "eq.Scalar": self._binary_op(relax.op.equal, operator.eq),
            "eq.Tensor": self._binary_op(relax.op.equal, operator.eq),
            "floor_divide.default": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "fmod.Scalar": self._fmod,
            "fmod.Tensor": self._fmod,
            "logaddexp.default": self._binary_op(relax.op.log_add_exp, torch.logaddexp),
            "ge.Scalar": self._binary_op(relax.op.greater_equal, operator.ge),
            "ge.Tensor": self._binary_op(relax.op.greater_equal, operator.ge),
            "gt.Scalar": self._binary_op(relax.op.greater, operator.gt),
            "gt.Tensor": self._binary_op(relax.op.greater, operator.gt),
            "le.Scalar": self._binary_op(relax.op.less_equal, operator.le),
            "le.Tensor": self._binary_op(relax.op.less_equal, operator.le),
            "lt.Scalar": self._binary_op(relax.op.less, operator.lt),
            "lt.Tensor": self._binary_op(relax.op.less, operator.lt),
            "matmul.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max.other": self._binary_op(relax.op.maximum, max),
            "min.other": self._binary_op(relax.op.minimum, min),
            "max.default": self._unary_op(relax.op.max),
            "min.default": self._unary_op(relax.op.min),
            "remainder.Tensor": self._binary_op(relax.op.floor_mod, operator.mod),
            "remainder.Scalar": self._binary_op(relax.op.floor_mod, operator.mod),
            "mul.Tensor": self._binary_op(relax.op.multiply, operator.mul),
            "mul_.Tensor": self._binary_op(relax.op.multiply, operator.mul),
            "ne.Tensor": self._binary_op(relax.op.not_equal, operator.ne),
            "ne.Scalar": self._binary_op(relax.op.not_equal, operator.ne),
            "outer.default": lambda node: self.block_builder.emit(
                relax.op.outer(self.env[node.args[0]], self.env[node.args[1]])
            ),
            "pow.Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relax.op.power, operator.pow),
            "sub.Tensor": self._binary_op(relax.op.subtract, operator.sub),
            "__and__.Tensor": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__and__.Scalar": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__or__.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__or__.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__xor__.Tensor": self._binary_op(relax.op.bitwise_xor, operator.xor),
            "__xor__.Scalar": self._binary_op(relax.op.bitwise_xor, operator.xor),
            # linear algebra
            "linalg_vector_norm.default": self._norm,
            # neural network
            "_native_batch_norm_legit_functional.default": self._batch_norm_legit_functional,
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "batch_norm.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool1d.default": self._adaptive_avg_pool1d,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "adaptive_avg_pool3d.default": self._adaptive_avg_pool3d,
            "addmm.default": self._addmm,
            "avg_pool1d.default": self._avg_pool1d,
            "avg_pool2d.default": self._avg_pool2d,
            "avg_pool3d.default": self._avg_pool3d,
            "baddbmm.default": self._baddbmm,
            "bmm.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "conv_transpose1d.default": self._conv_transpose1d,
            "conv_transpose2d.input": self._conv_transpose2d,
            "conv1d.default": self._conv1d,
            "conv2d.default": self._conv2d,
            "conv3d.default": self._conv3d,
            "cross_entropy_loss.default": self._cross_entropy_default,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "max_pool1d.default": self._max_pool1d,
            "max_pool2d.default": self._max_pool2d,
            "max_pool3d.default": self._max_pool3d,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            "upsample_bicubic2d.vec": self._upsample_bicubic2d,
            # statistical
            "mean.dim": self._mean,
            "prod.default": self._prod,
            "std.correction": self._std,
            "sum.default": self._sum,
            "sum.dim_IntList": self._sum,
            "var.correction": self._var,
            # search
            "argmax.default": self._argmax_argmin(relax.op.argmax),
            "argmin.default": self._argmax_argmin(relax.op.argmin),
            "where.self": self._where,
            # tensor manipulation
            "argsort.default": self._argsort,
            "broadcast_to.default": self._broadcast_to,
            "cat.default": self._cat,
            "chunk.default": self._chunk,
            "clamp.Tensor": self._clamp,
            "concat.default": self._cat,
            "copy_.default": self._copy_,
            "cumsum.default": self._cumsum,
            "cumprod.default": self._cumprod,
            "expand.default": self._expand,
            "expand_as.default": self._expand_as,
            "flatten.using_ints": self._flatten,
            "flip.default": self._flip,
            "gather.default": self._gather,
            "index.Tensor": self._index_tensor,
            "index_put_.default": self._index_put,
            "meshgrid.indexing": self._meshgrid,
            "meshgrid.default": self._meshgrid,
            "narrow.default": self._narrow,
            "permute.default": self._permute,
            "repeat.default": self._repeat,
            "roll.default": self._roll,
            "select.int": self._select,
            "slice.Tensor": self._slice,
            "slice_scatter.default": self._slice_scatter,
            "sort.default": self._sort,
            "split.Tensor": self._split,
            "split_with_sizes.default": self._split,
            "squeeze.default": self._squeeze,
            "squeeze.dim": self._squeeze,
            "stack.default": self._stack,
            "take.default": self._take,
            "tile.default": self._tile,
            "topk.default": self._topk,
            "transpose.int": self._transpose,
            "unflatten.int": self._unflatten,
            "unsqueeze.default": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view.default": self._reshape,
            "reshape.default": self._reshape,
            "reshape_as.default": self._reshape_as,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "arange.default": self._arange,
            "arange.start": self._arange,
            "arange.start_step": self._arange,
            "detach.default": self._detach,
            "detach_.default": self._detach,
            "contiguous.default": lambda node: self.env[node.args[0]],  # no-op
            "clone.default": lambda node: self.env[node.args[0]],
            "empty.memory_format": self._empty,
            "empty_like.default": self._empty_like,
            "eye.default": self._eye,
            "eye.m": self._eye,
            "fill.Scalar": self._fill,
            "fill_.Scalar": self._inplace_fill,
            "full.default": self._full,
            "full_like.default": self._full_like,
            "index_select.default": self._index_select,
            "lift_fresh_copy.default": self._to_copy,
            "linspace.default": self._linspace,
            "masked_fill.Scalar": self._masked_fill,
            "masked_fill_.Scalar": self._inplace_masked_fill,
            "new_ones.default": self._new_ones,
            "new_zeros.default": self._new_zeros,
            "one_hot.default": self._one_hot,
            "ones.default": self._ones,
            "ones_like.default": lambda node: self.block_builder.emit(
                relax.op.ones_like(self.env[node.args[0]])
            ),
            "zero_.default": self._zeros_inplace,
            "zeros.default": self._zeros,
            "zeros_like.default": self._zeros_like,
            # datatype
            "to.dtype": self._to,
            "to.dtype_layout": self._to,
            "type_as.default": self._type_as,
            # other
            "getitem": self._getitem,
            "item.default": self._item,
        }

    def create_input_vars(
        self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relax.Var], Dict[str, relax.Var]]:
        """Create relax input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        torch_symbol_to_relax_var: Dict[str, tvm.tir.Var] = {}

        for spec in exported_program.graph_signature.input_specs:
            name_hint = spec.arg.name
            if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
                torch_shape = exported_program.tensor_constants[spec.target].shape
                torch_dtype = exported_program.tensor_constants[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                for node in exported_program.graph.find_nodes(op="placeholder", target=spec.target):
                    if node.name == name_hint and "tensor_meta" in node.meta:
                        torch_shape = node.meta["tensor_meta"].shape
                        torch_dtype = node.meta["tensor_meta"].dtype
                        break
            else:
                # PARAMETER or BUFFER
                torch_shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype

            # TODO(mshr-h): Support range constraints
            relax_shape = [
                torch_symbol_to_relax_var.setdefault(str(s), tvm.tir.SizeVar(str(s), "int64"))
                if isinstance(s, torch.SymInt)
                else s
                for s in torch_shape
            ]
            dtype = self._convert_data_type(torch_dtype)

            relax_var = relax.Var(name_hint, relax.TensorStructInfo(relax_shape, dtype))
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relax_var
            else:
                parameters_buffers_constants[name_hint] = relax_var

        return parameters_buffers_constants, user_inputs

    def from_exported_program(
        self,
        exported_program: torch.export.ExportedProgram,
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a Relax program."""
        from torch import fx  # type: ignore

        # Create input variables.
        parameter_buffer_constant_vars, user_input_vars = self.create_input_vars(exported_program)
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        func_name = "main"
        func_attrs = {"num_input": len(user_input_vars)} if keep_params_as_input else None

        nodes: List[fx.Node] = exported_program.graph.nodes

        # Find all the missing function types
        self._check_unsupported_func_type(nodes)

        with self.block_builder.function(
            name=func_name, params=list(inputs_vars.values()).copy(), attrs=func_attrs
        ):
            output = None
            with self.block_builder.dataflow():

                # Translate the model.
                for node in nodes:
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Ignore sym input
                            continue

                        self.env[node] = inputs_vars[node.name]
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        assert isinstance(args[0], (tuple, relax.Tuple))

                        if unwrap_unit_return_tuple and len(args[0]) == 1:
                            output = self.block_builder.emit_output(args[0][0])
                        elif no_bind_return_tuple:
                            output = []
                            for ret in args[0]:
                                output.append(self.block_builder.emit_output(ret))
                        else:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = getattr(exported_program.graph_module, node.target)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        self.env[node] = self.convert_map[func_name](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        to_bind_parameters = ChainMap(
            OrderedDict(exported_program.named_buffers()), exported_program.constants
        )
        if not keep_params_as_input:
            to_bind_parameters = to_bind_parameters.new_child(
                OrderedDict(exported_program.named_parameters())
            )

        binding = {}
        for tensor_name, tensor_value in to_bind_parameters.items():
            # find relax var name from graph signature
            for spec in exported_program.graph_signature.input_specs:
                if tensor_name == spec.target:
                    bind_name = spec.arg.name
                    break
            binding[bind_name] = tvm.nd.from_dlpack(tensor_value.detach())

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [tvm.nd.from_dlpack(p.detach()) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
    exported_program: torch.export.ExportedProgram,
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
) -> tvm.IRModule:
    """Convert a PyTorch ExportedProgram to a Relax program

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The PyTorch ExportedProgram to convert.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.

    Examples
    --------
    Users can use the torch.export.export() to extract a torch.export.ExportedProgram
    from a PyTorch model. The following codes show how to convert a PyTorch model to
    a Relax program.

    .. code-block:: python

        # Import the importer.
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
        import torch
        from torch.export import export

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()

        # Use torch.export.export() to convert the PyTorch model into ExportedProgram.
        example_args = (torch.rand(128, 10, dtype=torch.float32),)
        exported_program = export(torch_model, args=example_args)

        # Use the importer to import the ExportedProgram to Relax.
        mod: tvm.IRModule = from_exported_program(exported_program)
    """
    # decompose into Core ATen operators
    exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
    )
