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
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import fx
import tvm
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to Relax."""

    @staticmethod
    def _convert_pytorch_tensor_to_tvm(tensor_value: torch.Tensor) -> tvm.runtime.Tensor:
        """Convert a PyTorch tensor to TVM tensor, handling sparse tensors.

        Parameters
        ----------
        tensor_value : torch.Tensor
            The PyTorch tensor to convert.

        Returns
        -------
        tvm.runtime.Tensor
            The converted TVM tensor.
        """
        # PyTorch sparse tensors (layout != torch.strided) must be converted to dense.
        if tensor_value.layout != torch.strided:
            tensor_to_convert = tensor_value.to_dense()
        else:
            tensor_to_convert = tensor_value
        tensor_detached = tensor_to_convert.detach()

        # Try DLPack conversion first (faster)
        try:
            return tvm.runtime.from_dlpack(tensor_detached)
        except (RuntimeError, BufferError):
            # Fallback: convert to numpy and then to TVM tensor
            # This handles cases where DLPack conversion fails
            tensor_cpu = tensor_detached.cpu().contiguous()
            return tvm.runtime.tensor(tensor_cpu.numpy())

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

    def _sqrt(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = x.struct_info.dtype

        # Check if input is integer type and convert to float32 if needed
        if dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
            x = self.block_builder.emit(relax.op.astype(x, "float32"))

        return self.block_builder.emit(relax.op.sqrt(x))

    def _rsqrt(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dtype = x.struct_info.dtype

        # Check if input is integer type and convert to float32 if needed
        if dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
            x = self.block_builder.emit(relax.op.astype(x, "float32"))

        return self.block_builder.emit(relax.op.rsqrt(x))

    ########## Neural Network ##########

    def _batch_norm(self, node: fx.Node, training: bool, return_tuple: bool = False) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))

        # After torch.export decomposition, batch_norm shows up as
        # _native_batch_norm_legit_* with signature (x, weight, bias, mean, var, momentum, eps).
        target_name = getattr(node.target, "__name__", "")
        if target_name.startswith("_native_batch_norm_legit_no_training"):
            momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
            eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)
            training = False
        elif target_name.startswith("_native_batch_norm_legit_functional"):
            momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
            eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)
            training = True
        else:
            ignore_running_stats = (
                node.args[5] if len(node.args) > 5 else node.kwargs.get("track_running_stats", True)
            )
            track_running_stats = not ignore_running_stats
            momentum = node.args[6] if len(node.args) > 6 else node.kwargs.get("momentum", 0.1)
            eps = node.args[7] if len(node.args) > 7 else node.kwargs.get("eps", 1e-05)

            if track_running_stats:
                training = True

        bn_result = self.block_builder.emit(
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
            )
        )

        if return_tuple:
            return bn_result
        else:
            # Return only the output tensor (for backward compatibility)
            return self.block_builder.emit(bn_result[0])

    def _batch_norm_legit_functional(self, node: fx.Node) -> relax.Var:
        # This method is called for batch_norm in training mode
        bn_tuple = self._batch_norm(node, training=True, return_tuple=True)

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype

        output = self.block_builder.emit(bn_tuple[0])
        new_running_mean = self.block_builder.emit(bn_tuple[1])
        reserve = self.block_builder.emit(relax.op.zeros(relax.ShapeExpr([channel]), dtype))

        return self.block_builder.emit(
            relax.Tuple([output, new_running_mean, reserve, reserve, reserve])
        )

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        return self._batch_norm(node, training=False, return_tuple=False)

    def _batch_norm_legit_no_stats(self, node: fx.Node) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        eps = node.args[5] if len(node.args) > 5 else node.kwargs.get("eps", 1e-05)

        # Determine axes for instance norm (all spatial dimensions after channel)
        dim = len(self.shape_of(x))
        axes = list(range(2, dim))

        return self.block_builder.emit(
            relax.op.nn.instance_norm(
                x,
                weight,
                bias,
                channel_axis=1,
                axes=axes,
                epsilon=eps,
            )
        )

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

    def _native_group_norm(self, node: fx.Node) -> relax.Var:
        # native_group_norm signature: (input, weight, bias, N, C, HxW, group, eps)
        x = self.env[node.args[0]]
        gamma = self.env.get(node.args[1], None) if len(node.args) > 1 else None
        beta = self.env.get(node.args[2], None) if len(node.args) > 2 else None
        # args[3] = N (batch size), args[4] = C (channels), args[5] = HxW (spatial size)
        num_groups = node.args[6] if len(node.args) > 6 else 1
        eps = node.args[7] if len(node.args) > 7 else 1e-05

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

    def _native_layer_norm(self, node: fx.Node) -> relax.Var:
        # native_layer_norm signature: (input, normalized_shape, weight, bias, eps)
        x = self.env[node.args[0]]
        normalized_shape = node.args[1]
        gamma = self.env.get(node.args[2], None) if len(node.args) > 2 else None
        beta = self.env.get(node.args[3], None) if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

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

    def _upsample_bilinear2d_aa(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", False)
        )
        scale_factor = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factors", None)
        )

        # Note: TVM's resize2d doesn't have explicit antialias support.
        # For upsampling, antialiasing has minimal effect, so we use regular bilinear.
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
            # PyTorch export passes scale_factor as either a scalar or a list/tuple
            # (e.g., [2.0, 3.0] for different H and W scaling).
            # Pass it as-is to _upsample_impl which handles both cases correctly.
            scale_factor = (
                node.args[2] if len(node.args) > 2 else node.kwargs.get("scale_factor", 1)
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
            # PyTorch export passes scale_factor as either a scalar or a list/tuple.
            # Pass it as-is to _upsample_impl which handles both cases correctly.
            scale_factor = (
                node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", 1)
            )

        return self._upsample_impl(
            x,
            size=size,
            scale_factor=scale_factor,
            method="cubic",
            align_corners=align_corners,
        )

    def _lstm_cell_unroll(
        self,
        input_reshaped,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        h_prev,
        c_prev,
        seq_len,
        hidden_size,
        reverse=False,
    ):
        """Unroll LSTM cells for a single direction."""
        weight_ih_t = self.block_builder.emit(relax.op.permute_dims(weight_ih, axes=[1, 0]))
        weight_hh_t = self.block_builder.emit(relax.op.permute_dims(weight_hh, axes=[1, 0]))
        outputs = []
        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in time_steps:
            x_t = self.block_builder.emit(
                relax.op.take(input_reshaped, relax.const(t, "int64"), axis=0, mode="clip")
            )
            ih_gates = self.block_builder.emit(relax.op.linear_algebra.matmul(x_t, weight_ih_t))
            hh_gates = self.block_builder.emit(relax.op.linear_algebra.matmul(h_prev, weight_hh_t))

            gates = self.block_builder.emit(relax.op.add(ih_gates, hh_gates))
            if bias_ih is not None:
                gates = self.block_builder.emit(relax.op.add(gates, bias_ih))
            if bias_hh is not None:
                gates = self.block_builder.emit(relax.op.add(gates, bias_hh))

            i_gate = self.block_builder.emit(
                relax.op.strided_slice(gates, axes=[1], begin=[0], end=[hidden_size])
            )
            f_gate = self.block_builder.emit(
                relax.op.strided_slice(gates, axes=[1], begin=[hidden_size], end=[2 * hidden_size])
            )
            g_gate = self.block_builder.emit(
                relax.op.strided_slice(
                    gates, axes=[1], begin=[2 * hidden_size], end=[3 * hidden_size]
                )
            )
            o_gate = self.block_builder.emit(
                relax.op.strided_slice(
                    gates, axes=[1], begin=[3 * hidden_size], end=[4 * hidden_size]
                )
            )

            i_t = self.block_builder.emit(relax.op.sigmoid(i_gate))
            f_t = self.block_builder.emit(relax.op.sigmoid(f_gate))
            g_t = self.block_builder.emit(relax.op.tanh(g_gate))
            o_t = self.block_builder.emit(relax.op.sigmoid(o_gate))

            c_t = self.block_builder.emit(
                relax.op.add(relax.op.multiply(f_t, c_prev), relax.op.multiply(i_t, g_t))
            )
            h_t = self.block_builder.emit(relax.op.multiply(o_t, relax.op.tanh(c_t)))

            outputs.append(h_t)
            h_prev = h_t
            c_prev = c_t

        if reverse:
            outputs = outputs[::-1]

        output = self.block_builder.emit(relax.op.stack(outputs, axis=0))
        return output

    def _lstm(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        hx = args[1] if len(args) > 1 else None
        params = args[2] if len(args) > 2 else None
        has_biases = args[3] if len(args) > 3 else True
        num_layers = args[4] if len(args) > 4 else 1
        bidirectional = args[7] if len(args) > 7 else False
        batch_first = args[8] if len(args) > 8 else False

        if num_layers > 1:
            raise NotImplementedError("Multi-layer LSTM is not yet supported")

        input_shape = self.shape_of(input_tensor)
        if batch_first:
            batch_size, seq_len, input_size = input_shape
        else:
            seq_len, batch_size, input_size = input_shape

        seq_len = int(seq_len) if isinstance(seq_len, tvm.tir.IntImm) else seq_len
        batch_size = int(batch_size) if isinstance(batch_size, tvm.tir.IntImm) else batch_size
        input_size = int(input_size) if isinstance(input_size, tvm.tir.IntImm) else input_size
        # Extract hidden size from the LSTM parameters
        # The parameters are: [weight_ih, weight_hh, bias_ih, bias_hh]
        # weight_ih shape: (4 * hidden_size, input_size)
        # weight_hh shape: (4 * hidden_size, hidden_size)
        if params and len(params) >= 2:
            # Extract hidden size from weight dimensions
            # weight_ih has shape (4 * hidden_size, input_size)
            weight_ih_shape = self.shape_of(params[0])
            hidden_size = weight_ih_shape[0] // 4
        else:
            # Fallback to a default hidden size
            hidden_size = 16
        # Implement actual LSTM computation using  Relax operations
        # LSTM equations:
        # i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
        # f_t = sigmoid(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
        # g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
        # o_t = sigmoid(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
        # c_t = f_t * c_{t-1} + i_t * g_t
        # h_t = o_t * tanh(c_t)
        dtype = input_tensor.struct_info.dtype
        params_per_direction = 4 if has_biases else 2

        # Extract or create forward direction weights
        if params and len(params) >= 2:
            weight_ih_fwd = params[0]
            weight_hh_fwd = params[1]
            bias_ih_fwd = params[2] if has_biases and len(params) > 2 else None
            bias_hh_fwd = params[3] if has_biases and len(params) > 3 else None
        else:
            # Fallback: create zero weights
            weight_ih_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((4 * hidden_size, input_size)), dtype)
            )
            weight_hh_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((4 * hidden_size, hidden_size)), dtype)
            )
            bias_ih_fwd = None
            bias_hh_fwd = None

        # Extract or create backward direction weights if bidirectional
        if bidirectional:
            if params and len(params) >= params_per_direction * 2:
                weight_ih_bwd = params[params_per_direction]
                weight_hh_bwd = params[params_per_direction + 1]
                bias_ih_bwd = params[params_per_direction + 2] if has_biases else None
                bias_hh_bwd = params[params_per_direction + 3] if has_biases else None
            else:
                # Fallback: create zero weights
                weight_ih_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((4 * hidden_size, input_size)), dtype)
                )
                weight_hh_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((4 * hidden_size, hidden_size)), dtype)
                )
                bias_ih_bwd = None
                bias_hh_bwd = None
        else:
            weight_ih_bwd = None
            weight_hh_bwd = None
            bias_ih_bwd = None
            bias_hh_bwd = None

        if hx is not None and len(hx) >= 2:
            h_0, c_0 = hx[0], hx[1]
            h_prev_fwd = self.block_builder.emit(
                relax.op.take(h_0, relax.const(0, "int64"), axis=0, mode="clip")
            )
            c_prev_fwd = self.block_builder.emit(
                relax.op.take(c_0, relax.const(0, "int64"), axis=0, mode="clip")
            )
            if bidirectional:
                h_prev_bwd = self.block_builder.emit(
                    relax.op.take(h_0, relax.const(1, "int64"), axis=0, mode="clip")
                )
                c_prev_bwd = self.block_builder.emit(
                    relax.op.take(c_0, relax.const(1, "int64"), axis=0, mode="clip")
                )
            else:
                h_prev_bwd = None
                c_prev_bwd = None
        else:
            h_prev_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
            )
            c_prev_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
            )
            if bidirectional:
                h_prev_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
                )
                c_prev_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
                )
            else:
                h_prev_bwd = None
                c_prev_bwd = None

        input_reshaped = (
            self.block_builder.emit(relax.op.permute_dims(input_tensor, axes=[1, 0, 2]))
            if batch_first
            else input_tensor
        )

        output_fwd = self._lstm_cell_unroll(
            input_reshaped,
            weight_ih_fwd,
            weight_hh_fwd,
            bias_ih_fwd,
            bias_hh_fwd,
            h_prev_fwd,
            c_prev_fwd,
            seq_len,
            hidden_size,
            reverse=False,
        )

        if bidirectional:
            output_bwd = self._lstm_cell_unroll(
                input_reshaped,
                weight_ih_bwd,
                weight_hh_bwd,
                bias_ih_bwd,
                bias_hh_bwd,
                h_prev_bwd,
                c_prev_bwd,
                seq_len,
                hidden_size,
                reverse=True,
            )
            output = self.block_builder.emit(relax.op.concat([output_fwd, output_bwd], axis=2))
        else:
            output = output_fwd

        if batch_first:
            # (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
            output = self.block_builder.emit(relax.op.permute_dims(output, axes=[1, 0, 2]))
        return output

    def _gru_cell_unroll(
        self,
        input_reshaped,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        h_prev,
        seq_len,
        hidden_size,
        dtype,
        reverse=False,
    ):
        """Unroll GRU cells for a single direction."""
        gate_size = hidden_size

        # Split weights by gates: PyTorch GRU gate order: reset, update, new (r, z, n)
        # Reset gate weights
        weight_ih_r = self.block_builder.emit(
            relax.op.strided_slice(weight_ih, axes=[0], begin=[0], end=[gate_size])
        )
        weight_hh_r = self.block_builder.emit(
            relax.op.strided_slice(weight_hh, axes=[0], begin=[0], end=[gate_size])
        )

        # Update gate weights
        weight_ih_z = self.block_builder.emit(
            relax.op.strided_slice(weight_ih, axes=[0], begin=[gate_size], end=[2 * gate_size])
        )
        weight_hh_z = self.block_builder.emit(
            relax.op.strided_slice(weight_hh, axes=[0], begin=[gate_size], end=[2 * gate_size])
        )

        # New gate weights
        weight_ih_n = self.block_builder.emit(
            relax.op.strided_slice(weight_ih, axes=[0], begin=[2 * gate_size], end=[3 * gate_size])
        )
        weight_hh_n = self.block_builder.emit(
            relax.op.strided_slice(weight_hh, axes=[0], begin=[2 * gate_size], end=[3 * gate_size])
        )

        # Transpose weights for matmul
        weight_ih_r_t = self.block_builder.emit(relax.op.permute_dims(weight_ih_r, axes=[1, 0]))
        weight_hh_r_t = self.block_builder.emit(relax.op.permute_dims(weight_hh_r, axes=[1, 0]))
        weight_ih_z_t = self.block_builder.emit(relax.op.permute_dims(weight_ih_z, axes=[1, 0]))
        weight_hh_z_t = self.block_builder.emit(relax.op.permute_dims(weight_hh_z, axes=[1, 0]))
        weight_ih_n_t = self.block_builder.emit(relax.op.permute_dims(weight_ih_n, axes=[1, 0]))
        weight_hh_n_t = self.block_builder.emit(relax.op.permute_dims(weight_hh_n, axes=[1, 0]))

        outputs = []
        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in time_steps:
            # Get input at time t: (batch_size, input_size)
            x_t = self.block_builder.emit(
                relax.op.take(input_reshaped, relax.const(t, "int64"), axis=0, mode="clip")
            )

            # Compute reset gate: r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
            r_ih = self.block_builder.emit(relax.op.linear_algebra.matmul(x_t, weight_ih_r_t))
            r_hh = self.block_builder.emit(relax.op.linear_algebra.matmul(h_prev, weight_hh_r_t))
            if bias_ih is not None and bias_hh is not None:
                bias_ih_r = self.block_builder.emit(
                    relax.op.strided_slice(bias_ih, axes=[0], begin=[0], end=[gate_size])
                )
                bias_hh_r = self.block_builder.emit(
                    relax.op.strided_slice(bias_hh, axes=[0], begin=[0], end=[gate_size])
                )
                r_t = self.block_builder.emit(
                    relax.op.sigmoid(
                        relax.op.add(relax.op.add(relax.op.add(r_ih, bias_ih_r), r_hh), bias_hh_r)
                    )
                )
            else:
                r_t = self.block_builder.emit(relax.op.sigmoid(relax.op.add(r_ih, r_hh)))

            # Compute update gate: z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
            z_ih = self.block_builder.emit(relax.op.linear_algebra.matmul(x_t, weight_ih_z_t))
            z_hh = self.block_builder.emit(relax.op.linear_algebra.matmul(h_prev, weight_hh_z_t))
            if bias_ih is not None and bias_hh is not None:
                bias_ih_z = self.block_builder.emit(
                    relax.op.strided_slice(
                        bias_ih, axes=[0], begin=[gate_size], end=[2 * gate_size]
                    )
                )
                bias_hh_z = self.block_builder.emit(
                    relax.op.strided_slice(
                        bias_hh, axes=[0], begin=[gate_size], end=[2 * gate_size]
                    )
                )
                z_t = self.block_builder.emit(
                    relax.op.sigmoid(
                        relax.op.add(relax.op.add(relax.op.add(z_ih, bias_ih_z), z_hh), bias_hh_z)
                    )
                )
            else:
                z_t = self.block_builder.emit(relax.op.sigmoid(relax.op.add(z_ih, z_hh)))

            # Compute new gate: n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
            n_ih = self.block_builder.emit(relax.op.linear_algebra.matmul(x_t, weight_ih_n_t))
            n_hh = self.block_builder.emit(relax.op.linear_algebra.matmul(h_prev, weight_hh_n_t))
            if bias_ih is not None and bias_hh is not None:
                bias_ih_n = self.block_builder.emit(
                    relax.op.strided_slice(
                        bias_ih, axes=[0], begin=[2 * gate_size], end=[3 * gate_size]
                    )
                )
                bias_hh_n = self.block_builder.emit(
                    relax.op.strided_slice(
                        bias_hh, axes=[0], begin=[2 * gate_size], end=[3 * gate_size]
                    )
                )
                n_t = self.block_builder.emit(
                    relax.op.tanh(
                        relax.op.add(
                            relax.op.add(n_ih, bias_ih_n),
                            relax.op.multiply(r_t, relax.op.add(n_hh, bias_hh_n)),
                        )
                    )
                )
            else:
                n_t = self.block_builder.emit(
                    relax.op.tanh(relax.op.add(n_ih, relax.op.multiply(r_t, n_hh)))
                )

            # Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
            one_minus_z = self.block_builder.emit(relax.op.subtract(relax.const(1.0, dtype), z_t))
            h_t = self.block_builder.emit(
                relax.op.add(relax.op.multiply(one_minus_z, n_t), relax.op.multiply(z_t, h_prev))
            )

            outputs.append(h_t)
            h_prev = h_t

        if reverse:
            outputs = outputs[::-1]

        output = self.block_builder.emit(relax.op.stack(outputs, axis=0))
        return output

    def _gru(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        hx = args[1] if len(args) > 1 else None
        params = args[2] if len(args) > 2 else None
        has_biases = args[3] if len(args) > 3 else True
        num_layers = args[4] if len(args) > 4 else 1
        _dropout = args[5] if len(args) > 5 else 0.0  # Not used in inference
        _train = args[6] if len(args) > 6 else False  # Not used in inference
        bidirectional = args[7] if len(args) > 7 else False
        batch_first = args[8] if len(args) > 8 else False

        if num_layers > 1:
            raise NotImplementedError("Multi-layer GRU is not yet supported")

        input_shape = self.shape_of(input_tensor)
        if batch_first:
            batch_size, seq_len, input_size = input_shape
        else:
            seq_len, batch_size, input_size = input_shape

        seq_len = int(seq_len) if isinstance(seq_len, tvm.tir.IntImm) else seq_len
        batch_size = int(batch_size) if isinstance(batch_size, tvm.tir.IntImm) else batch_size
        input_size = int(input_size) if isinstance(input_size, tvm.tir.IntImm) else input_size

        # Extract hidden size from parameters
        # For bidirectional: params has weights for both directions
        # params_per_direction = 4 if has_biases else 2 (weight_ih, weight_hh, [bias_ih, bias_hh])
        params_per_direction = 4 if has_biases else 2

        if params and len(params) >= 2:
            # Extract hidden size from weight dimensions
            # weight_ih has shape (3 * hidden_size, input_size)
            weight_ih_shape = self.shape_of(params[0])
            hidden_size = weight_ih_shape[0] // 3  # 3 gates: reset, update, new
        else:
            # Fallback to a default hidden size
            hidden_size = 16

        dtype = input_tensor.struct_info.dtype

        # Extract forward direction weights
        if params and len(params) >= params_per_direction:
            weight_ih_fwd = params[0]
            weight_hh_fwd = params[1]
            bias_ih_fwd = params[2] if has_biases else None
            bias_hh_fwd = params[3] if has_biases else None
        else:
            # Fallback: create zero weights
            weight_ih_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((3 * hidden_size, input_size)), dtype)
            )
            weight_hh_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((3 * hidden_size, hidden_size)), dtype)
            )
            bias_ih_fwd = None
            bias_hh_fwd = None

        # Extract or create backward direction weights if bidirectional
        if bidirectional:
            if params and len(params) >= params_per_direction * 2:
                weight_ih_bwd = params[params_per_direction]
                weight_hh_bwd = params[params_per_direction + 1]
                bias_ih_bwd = params[params_per_direction + 2] if has_biases else None
                bias_hh_bwd = params[params_per_direction + 3] if has_biases else None
            else:
                # Fallback: create zero weights
                weight_ih_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((3 * hidden_size, input_size)), dtype)
                )
                weight_hh_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((3 * hidden_size, hidden_size)), dtype)
                )
                bias_ih_bwd = None
                bias_hh_bwd = None
        else:
            weight_ih_bwd = None
            weight_hh_bwd = None
            bias_ih_bwd = None
            bias_hh_bwd = None

        # Initialize hidden states
        if hx is not None:
            h_prev_fwd = self.block_builder.emit(
                relax.op.take(hx, relax.const(0, "int64"), axis=0, mode="clip")
            )
            if bidirectional:
                h_prev_bwd = self.block_builder.emit(
                    relax.op.take(hx, relax.const(1, "int64"), axis=0, mode="clip")
                )
            else:
                h_prev_bwd = None
        else:
            h_prev_fwd = self.block_builder.emit(
                relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
            )
            if bidirectional:
                h_prev_bwd = self.block_builder.emit(
                    relax.op.zeros(relax.ShapeExpr((batch_size, hidden_size)), dtype)
                )
            else:
                h_prev_bwd = None

        # Reshape input for processing
        input_reshaped = (
            self.block_builder.emit(relax.op.permute_dims(input_tensor, axes=[1, 0, 2]))
            if batch_first
            else input_tensor
        )

        # Process forward direction
        output_fwd = self._gru_cell_unroll(
            input_reshaped,
            weight_ih_fwd,
            weight_hh_fwd,
            bias_ih_fwd,
            bias_hh_fwd,
            h_prev_fwd,
            seq_len,
            hidden_size,
            dtype,
            reverse=False,
        )

        # Process backward direction if bidirectional
        if bidirectional:
            output_bwd = self._gru_cell_unroll(
                input_reshaped,
                weight_ih_bwd,
                weight_hh_bwd,
                bias_ih_bwd,
                bias_hh_bwd,
                h_prev_bwd,
                seq_len,
                hidden_size,
                dtype,
                reverse=True,
            )
            # Concatenate forward and backward outputs along feature dimension
            output = self.block_builder.emit(relax.op.concat([output_fwd, output_bwd], axis=2))
        else:
            output = output_fwd

        # Reshape back to batch_first if needed
        if batch_first:
            # (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
            output = self.block_builder.emit(relax.op.permute_dims(output, axes=[1, 0, 2]))

        return output

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
        import sys

        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else None
        end_val = node.args[3] if len(node.args) > 3 else None
        step = node.args[4] if len(node.args) > 4 else 1

        if start is None:
            start = 0
        if end_val is None:
            end_val = sys.maxsize

        axes = [dim]
        begin = [start]
        end = [end_val]
        stride = [step]
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

    def _hamming_window(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)

        window_size = args[0]
        periodic = args[1] if len(args) > 1 else True
        alpha = args[2] if len(args) > 2 else 0.54
        beta = args[3] if len(args) > 3 else 0.46
        dtype = node.kwargs.get("dtype", "float")
        dtype = self._convert_data_type(dtype)

        return self.block_builder.emit(
            relax.op.hamming_window(window_size, periodic, alpha, beta, dtype)
        )

    def _zeros(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        size = relax.ShapeExpr(args[0] if isinstance(args[0], (list, tuple)) else (args[0],))
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(relax.op.zeros(size, dtype))

    def _sparse_mm(self, node: fx.Node) -> relax.Var:
        """Handle sparse matrix multiplication by converting sparse tensor to dense."""
        args = self.retrieve_args(node)
        sparse_input = args[0]
        dense_input = args[1]
        # Convert sparse tensor to dense if needed
        # Note: sparse_input should already be converted to dense in _convert_pytorch_tensor_to_tvm
        # Use regular matrix multiplication
        return self.block_builder.emit(
            relax.op.linear_algebra.matmul(sparse_input, dense_input, out_dtype="float32")
        )

    def _sparse_addmm(self, node: fx.Node) -> relax.Var:
        """Handle sparse addmm (beta * input + alpha * sparse_mm(mat1, mat2))."""
        args = self.retrieve_args(node)
        input_tensor = args[0]  # beta * input
        sparse_mat1 = args[1]  # sparse matrix
        dense_mat2 = args[2]  # dense matrix
        alpha = node.kwargs.get("alpha", 1.0)
        beta = node.kwargs.get("beta", 1.0)

        # Convert sparse tensor to dense if needed
        # Note: sparse_mat1 should already be converted to dense in _convert_pytorch_tensor_to_tvm
        # Compute alpha * sparse_mm(mat1, mat2)
        matmul_result = self.block_builder.emit(
            relax.op.linear_algebra.matmul(sparse_mat1, dense_mat2, out_dtype="float32")
        )

        if alpha != 1.0:
            alpha_const = relax.const(alpha, matmul_result.struct_info.dtype)
            matmul_result = self.block_builder.emit(relax.op.multiply(matmul_result, alpha_const))

        # Compute beta * input + alpha * matmul_result
        if beta != 0.0:
            if beta != 1.0:
                beta_const = relax.const(beta, input_tensor.struct_info.dtype)
                input_scaled = self.block_builder.emit(relax.op.multiply(input_tensor, beta_const))
            else:
                input_scaled = input_tensor
            return self.block_builder.emit(relax.op.add(input_scaled, matmul_result))
        else:
            return matmul_result

    def _grid_sampler_2d(self, node: fx.Node) -> relax.Var:
        """Convert torch.nn.functional.grid_sample to relax.op.image.grid_sample."""
        args = self.retrieve_args(node)
        data = args[0]
        grid = args[1]
        interp_mode = args[2] if len(args) > 2 else 0
        pad_mode = args[3] if len(args) > 3 else 0
        align_corners = args[4] if len(args) > 4 else False

        interp_map = {0: "bilinear", 1: "nearest", 2: "bicubic"}
        pad_map = {0: "zeros", 1: "border", 2: "reflection"}

        method = interp_map.get(interp_mode, "bilinear")
        padding_mode = pad_map.get(pad_mode, "zeros")

        return self.block_builder.emit(
            relax.op.image.grid_sample(
                data,
                grid,
                method=method,
                layout="NCHW",
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
        )

    def _scalar_tensor(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        scalar_value = args[0]
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(relax.const(scalar_value, dtype))

    def _instance_norm(self, node: fx.Node):
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        gamma = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        beta = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        eps = node.args[4] if node.args[4] else 1e-05
        channel_axis = 1
        dim = len(self.shape_of(x))

        return self.block_builder.emit(
            relax.op.nn.instance_norm(
                x,
                gamma,
                beta,
                channel_axis=channel_axis,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _exponential(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.zeros_like(x))

    def _max_dim(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)

        topk_res = self.block_builder.emit(
            relax.op.topk(x, k=1, axis=dim, largest=True, ret_type="both", dtype="int64")
        )

        values = topk_res[0]
        indices = topk_res[1]

        if not keepdim:
            values = self.block_builder.emit(relax.op.squeeze(values, axis=[dim]))
            indices = self.block_builder.emit(relax.op.squeeze(indices, axis=[dim]))

        return self.block_builder.emit(relax.Tuple([values, indices]))

    def _alias(self, node: fx.Node) -> relax.Var:
        return self.env[node.args[0]]

    def _scatter_value(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = self.env[node.args[2]]
        value = node.args[3]

        value_const = relax.const(value, x.struct_info.dtype)
        src = self.block_builder.emit(relax.op.broadcast_to(value_const, self.shape_of(index)))

        return self.block_builder.emit(relax.op.scatter_elements(x, index, src, axis=dim))

    def _as_strided(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        size = args[1]
        stride = args[2]
        storage_offset = args[3] if len(args) > 3 else node.kwargs.get("storage_offset", 0)

        assert storage_offset == 0, "as_strided with non-zero storage_offset is not supported yet"

        # Only handle view-like cases where the provided strides align with a contiguous layout.
        can_check = all(isinstance(dim, (int, tvm.tir.IntImm)) for dim in size) and all(
            isinstance(st, (int, tvm.tir.IntImm)) for st in stride
        )
        if can_check:
            expected_stride = []
            running = 1
            for dim in reversed(size):
                dim_int = int(dim)
                expected_stride.insert(0, running)
                running *= dim_int

            for dim, st, exp in zip(size, stride, expected_stride):
                dim_int = int(dim)
                if dim_int != 1 and int(st) != exp:
                    raise AssertionError(
                        f"as_strided with non-contiguous stride {stride} for"
                        f"size {size} is not supported"
                    )

        return self.block_builder.emit(relax.op.reshape(x, size))

    ########## Symbolic Shape Constraints ##########

    def _symbolic_comparison(self, _: fx.Node) -> relax.Expr:
        return self.block_builder.emit(relax.const(True, dtype="bool"))

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
            "native_dropout.default": lambda node: self.env[node.args[0]],
            "elu.default": self._elu,
            "erf.default": self._unary_op(relax.op.erf),
            "exp.default": self._unary_op(relax.op.exp),
            "exponential.default": self._exponential,
            "expm1.default": lambda node: self.block_builder.emit(
                relax.op.subtract(
                    relax.op.exp(self.env[node.args[0]]),
                    relax.const(1.0, self.env[node.args[0]].struct_info.dtype),
                )
            ),
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
            "logical_and.default": self._binary_op(relax.op.logical_and, operator.and_),
            "log_softmax.int": self._log_softmax,
            "_log_softmax.default": self._log_softmax,
            "neg.default": self._unary_op(relax.op.negative),
            "pad.default": self._pad,
            "constant_pad_nd.default": self._constant_pad_nd,
            "copy.default": self._copy_,
            "pixel_shuffle.default": self._pixel_shuffle,
            "prelu.default": self._prelu,
            "reciprocal.default": self._reciprocal,
            "relu.default": self._unary_op(relax.op.nn.relu),
            "relu_.default": self._unary_op(relax.op.nn.relu),
            "relu6.default": self._unary_op(relax.op.nn.relu6),
            "relu6_.default": self._unary_op(relax.op.nn.relu6),
            "round.default": self._round,
            "rsqrt.default": self._rsqrt,
            "scalar_tensor.default": self._scalar_tensor,
            "scatter.value": self._scatter_value,
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
            "_softmax.default": self._softmax,
            "softplus.default": self._softplus,
            "softshrink.default": self._softshrink,
            "softsign.default": self._softsign,
            "sqrt.default": self._sqrt,
            "square.default": self._unary_op(relax.op.square),
            "tan.default": self._unary_op(relax.op.tan),
            "tanh.default": self._unary_op(relax.op.tanh),
            "tril.default": self._tril_triu(relax.op.tril),
            "triu.default": self._tril_triu(relax.op.triu),
            "trunc.default": self._unary_op(relax.op.trunc),
            # binary
            "add.Tensor": self._binary_op(relax.op.add, operator.add),
            "add.Scalar": self._binary_op(relax.op.add, operator.add),
            "add_.Tensor": self._binary_op(relax.op.add, operator.add),
            "bitwise_and.Tensor": self._binary_op(relax.op.bitwise_and, operator.and_),
            "bitwise_and.Scalar": self._binary_op(relax.op.bitwise_and, operator.and_),
            "bitwise_or_.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_or.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_xor.Tensor": self._binary_op(relax.op.bitwise_xor, operator.xor),
            "bitwise_xor.Scalar": self._binary_op(relax.op.bitwise_xor, operator.xor),
            "bitwise_or_.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "bitwise_or.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "div.Scalar": self._binary_op(relax.op.divide, operator.truediv),
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
            "mm.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max.other": self._binary_op(relax.op.maximum, max),
            "min.other": self._binary_op(relax.op.minimum, min),
            "max.default": self._unary_op(relax.op.max),
            "min.default": self._unary_op(relax.op.min),
            "maximum.default": self._binary_op(relax.op.maximum, torch.maximum),
            "minimum.default": self._binary_op(relax.op.minimum, torch.minimum),
            "remainder.Tensor": self._binary_op(relax.op.floor_mod, operator.mod),
            "remainder.Scalar": self._binary_op(relax.op.floor_mod, operator.mod),
            "mul": self._binary_op(relax.op.multiply, operator.mul),
            "mul.Tensor": self._binary_op(relax.op.multiply, operator.mul),
            "mul.Scalar": self._binary_op(relax.op.multiply, operator.mul),
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
            "sub.Scalar": self._binary_op(relax.op.subtract, operator.sub),
            "__and__.Tensor": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__and__.Scalar": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__or__.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__or__.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__xor__.Tensor": self._binary_op(relax.op.bitwise_xor, operator.xor),
            "__xor__.Scalar": self._binary_op(relax.op.bitwise_xor, operator.xor),
            # linear algebra
            "linalg_vector_norm.default": self._norm,
            # neural network
            "_adaptive_avg_pool1d.default": self._adaptive_avg_pool1d,
            "_adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "_adaptive_avg_pool3d.default": self._adaptive_avg_pool3d,
            "_native_batch_norm_legit_functional.default": self._batch_norm_legit_functional,
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "_native_batch_norm_legit.no_stats": self._batch_norm_legit_no_stats,
            "batch_norm.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool1d.default": self._adaptive_avg_pool1d,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "adaptive_avg_pool3d.default": self._adaptive_avg_pool3d,
            "addmm.default": self._addmm,
            "_sparse_mm.default": self._sparse_mm,
            "_sparse_addmm.default": self._sparse_addmm,
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
            "convolution.default": self._convolution,
            "cross_entropy_loss.default": self._cross_entropy_default,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "instance_norm.default": self._instance_norm,
            "native_group_norm.default": self._native_group_norm,
            "layer_norm.default": self._layer_norm,
            "native_layer_norm.default": self._native_layer_norm,
            "linear.default": self._linear,
            "lstm.input": self._lstm,
            "gru.input": self._gru,
            "max_pool1d.default": self._max_pool1d,
            "max_pool2d.default": self._max_pool2d,
            "max_pool2d_with_indices.default": self._max_pool2d_with_indices,
            "max_pool3d.default": self._max_pool3d,
            "max_pool3d_with_indices.default": self._max_pool3d_with_indices,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "_upsample_bilinear2d_aa.default": self._upsample_bilinear2d_aa,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            "upsample_bicubic2d.vec": self._upsample_bicubic2d,
            # statistical
            "any.dim": self._any,
            "any.dims": self._any,
            "mean.dim": self._mean,
            "prod.default": self._prod,
            "std.correction": self._std,
            "sum.default": self._sum,
            "sum.dim_IntList": self._sum,
            "var.correction": self._var,
            "max.dim": self._max_dim,
            # search
            "argmax.default": self._argmax_argmin(relax.op.argmax),
            "argmin.default": self._argmax_argmin(relax.op.argmin),
            "where.self": self._where,
            "bucketize.Tensor": self._bucketize,
            # tensor manipulation
            "argsort.default": self._argsort,
            "alias.default": self._alias,
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
            "index_put.default": self._index_put,
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
            "squeeze.dims": self._squeeze,
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
            "as_strided.default": self._as_strided,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "arange.default": self._arange,
            "arange.start": self._arange,
            "arange.start_step": self._arange,
            "detach.default": self._detach,
            "detach_.default": self._detach,
            "contiguous.default": lambda node: self.env[node.args[0]],  # no-op
            "clone.default": lambda node: self.env[node.args[0]],
            "bernoulli.p": lambda node: self.env[node.args[0]],  # Dropout: just return input
            "_assert_tensor_metadata.default": lambda node: self.env[
                node.args[0]
            ],  # metadata assertion: no-op
            "empty.default": self._empty,
            "empty.memory_format": self._empty,
            "empty_permuted.default": self._empty,  # Similar to empty with permuted layout
            "empty_like.default": self._empty_like,
            "eye.default": self._eye,
            "eye.m": self._eye,
            "fill.Scalar": self._fill,
            "fill_.Scalar": self._inplace_fill,
            "full.default": self._full,
            "full_like.default": self._full_like,
            "hamming_window.periodic": self._hamming_window,
            "hamming_window.periodic_alpha": self._hamming_window,
            "hamming_window.periodic_alpha_beta": self._hamming_window,
            "hamming_window.default": self._hamming_window,
            "index_select.default": self._index_select,
            "lift_fresh_copy.default": self._to_copy,
            "linspace.default": self._linspace,
            "masked_fill.Scalar": self._masked_fill,
            "masked_fill_.Scalar": self._inplace_masked_fill,
            "masked_select.default": self._masked_select,
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
            "grid_sampler_2d.default": self._grid_sampler_2d,
            # datatype
            "to.dtype": self._to,
            "to.dtype_layout": self._to,
            "type_as.default": self._type_as,
            # other
            "getitem": self._getitem,
            "item.default": self._item,
            "sym_size.int": self._sym_size_int,
            "_local_scalar_dense.default": self._item,
            # symbolic shape constraints (no-ops for compilation)
            "sym_constrain_range_for_size.default": lambda node: self.env[node.args[0]],
            "_assert_scalar.default": lambda node: self.env[node.args[0]],
            "ge": self._symbolic_comparison,
            "le": self._symbolic_comparison,
        }

    def _process_derived_symbol(
        self, symbol, torch_symbol_to_relax_var: Dict[str, tvm.tir.Var]
    ) -> Tuple[str, Optional[tvm.tir.PrimExpr]]:
        """Process a sympy symbol to generate a descriptive name and TIR expression."""
        import sympy

        if isinstance(symbol, sympy.Symbol):
            return str(symbol), None

        if not isinstance(symbol, (sympy.Add, sympy.Mul)):
            return str(symbol), None

        tir_expr = None
        for arg in symbol.args:
            if isinstance(arg, sympy.Integer):
                term = tvm.tir.IntImm("int64", int(arg))
            elif isinstance(arg, sympy.Symbol):
                term = torch_symbol_to_relax_var.setdefault(
                    str(arg), tvm.tir.SizeVar(str(arg), "int64")
                )
            else:
                _, term = self._process_derived_symbol(arg, torch_symbol_to_relax_var)

            if term is None:
                return str(symbol), None

            if tir_expr is None:
                tir_expr = term
            elif isinstance(symbol, sympy.Mul):
                tir_expr = tir_expr * term
            elif isinstance(symbol, sympy.Add):
                tir_expr = tir_expr + term

        if isinstance(tir_expr, tvm.tir.Add):
            for const, var in [(tir_expr.a, tir_expr.b), (tir_expr.b, tir_expr.a)]:
                if isinstance(const, tvm.tir.IntImm) and isinstance(var, tvm.tir.Var):
                    return f"{var.name}___{const.value}", tir_expr

        if isinstance(tir_expr, tvm.tir.Mul):
            for const, var in [(tir_expr.a, tir_expr.b), (tir_expr.b, tir_expr.a)]:
                if isinstance(const, tvm.tir.IntImm) and isinstance(var, tvm.tir.Var):
                    return f"{var.name}_{const.value}", tir_expr

        return str(symbol), tir_expr

    def create_input_vars(
        self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relax.Var], Dict[str, relax.Var], Dict[str, Tuple[int, Optional[int]]]]:
        """Create relax input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        torch_symbol_to_relax_var: Dict[str, tvm.tir.Var] = {}
        range_constraints = {}

        if hasattr(exported_program, "range_constraints"):
            import math

            for symbol, value_range in exported_program.range_constraints.items():
                if hasattr(value_range, "lower") and hasattr(value_range, "upper"):
                    try:
                        # PyTorch uses int_oo (IntInfinity) for unbounded constraints
                        lower = int(value_range.lower)
                        upper = (
                            None if math.isinf(float(value_range.upper)) else int(value_range.upper)
                        )

                        symbol_name, _ = self._process_derived_symbol(
                            symbol, torch_symbol_to_relax_var
                        )
                        range_constraints[symbol_name] = (lower, upper)

                    except (OverflowError, AttributeError, TypeError):
                        continue

        named_buffers = OrderedDict(exported_program.named_buffers())
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
            elif spec.kind is torch.export.graph_signature.InputKind.BUFFER:
                torch_shape = named_buffers[spec.target].shape
                torch_dtype = named_buffers[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.PARAMETER:
                torch_shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype
            else:
                raise ValueError(f"Unsupported input kind: {spec.kind}")

            relax_shape = []
            for s in torch_shape:
                if isinstance(s, torch.SymInt):
                    sympy_node = s.node.expr if hasattr(s.node, "expr") else s.node
                    symbol_name, _ = self._process_derived_symbol(
                        sympy_node, torch_symbol_to_relax_var
                    )

                    size_var = torch_symbol_to_relax_var.setdefault(
                        symbol_name, tvm.tir.SizeVar(symbol_name, "int64")
                    )
                    relax_shape.append(size_var)
                else:
                    relax_shape.append(s)
            dtype = self._convert_data_type(torch_dtype)

            relax_var = relax.Var(name_hint, relax.TensorStructInfo(relax_shape, dtype))
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relax_var
            else:
                parameters_buffers_constants[name_hint] = relax_var

        return parameters_buffers_constants, user_inputs, range_constraints

    def from_exported_program(
        self,
        exported_program: torch.export.ExportedProgram,
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
        custom_convert_map: Optional[
            Dict[str, Callable[[fx.Node, BaseFXGraphImporter], relax.Var]]
        ],
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a Relax program."""

        # Update the conversion map with custom ops if provided.
        if custom_convert_map:
            custom_ops = set(custom_convert_map.keys())
            self.update_convert_map(custom_convert_map)
        else:
            custom_ops = set()

        # Create input variables.
        (
            parameter_buffer_constant_vars,
            user_input_vars,
            range_constraints,
        ) = self.create_input_vars(exported_program)
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        func_name = "main"
        func_attrs = {"num_input": len(user_input_vars)} if keep_params_as_input else {}
        if range_constraints:
            func_attrs["tir_var_lower_bound"] = {
                var_name: lower for var_name, (lower, _) in range_constraints.items()
            }

            upper_bounds = {
                var_name: upper
                for var_name, (_, upper) in range_constraints.items()
                if upper is not None
            }

            if upper_bounds:
                func_attrs["tir_var_upper_bound"] = upper_bounds

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
                        if func_name in custom_ops:
                            self.env[node] = self.convert_map[func_name](node, self)
                        else:
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
            binding[bind_name] = self._convert_pytorch_tensor_to_tvm(tensor_value)

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [self._convert_pytorch_tensor_to_tvm(p) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
    exported_program: torch.export.ExportedProgram,
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
    custom_convert_map: Optional[
        Dict[str, Callable[[fx.Node, BaseFXGraphImporter], relax.Var]]
    ] = None,
    run_ep_decomposition: bool = True,
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

    custom_convert_map : Dict[str, Callable[[fx.Node, BaseFXGraphImporter], relax.Var]]
        A custom op conversion map in the same format as ExportedProgramImporter.convert_map above

    run_ep_decomposition : bool
        A boolean flag indicating whether to run PyTorch's decomposition on the
        exported program before translation. When True, high-level operators will
        be decomposed into their constituent parts. Defaults to True.

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
    # Conditionally decompose into Core ATen operators
    if run_ep_decomposition:
        exported_program = exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
        custom_convert_map,
    )
