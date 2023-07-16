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
# pylint: disable=invalid-name,unused-argument
"""Default legalization function for perators to implement operaor gradients."""
import logging

from tvm import te, tir, topi
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


@register_legalize("relax.grad.no_grad")
def _no_grad(bb: BlockBuilder, call: Call) -> Expr:
    return call.args[0]


@register_legalize("relax.grad.start_checkpoint")
def _start_checkpoint(bb: BlockBuilder, call: Call) -> Expr:
    return call.args[0]


@register_legalize("relax.grad.end_checkpoint")
def _end_checkpoint(bb: BlockBuilder, call: Call) -> Expr:
    return call.args[0]


@register_legalize("relax.grad.nll_loss_backward")
def _grad_nll_loss_backward(bb: BlockBuilder, call: Call) -> Expr:
    # topi.sum don't support zero-dim x
    # we add support for that
    def topi_sum_extend(x):
        return x if x.ndim == 0 else topi.sum(x)

    def te_nll_loss_backward(output_grad, predictions, targets, weights, reduction, ignore_index):
        # handle ignore_index
        if ignore_index >= 0:
            weights = te.compute(
                weights.shape,
                lambda i: tir.Select(i == ignore_index, tir.const(0, weights.dtype), weights(i)),
                "weights_new",
            )

        all_weights = te.compute(targets.shape, lambda *i: weights(targets(*i)), "all_weights")

        # handle reduction
        if reduction == "sum":
            output_grad = topi.broadcast_to(output_grad, targets.shape)
        elif reduction == "mean":
            weight_sum = topi_sum_extend(all_weights)
            output_grad = topi.divide(topi.broadcast_to(output_grad, targets.shape), weight_sum)

        # handle no batch
        if predictions.ndim == 1:
            return te.compute(
                predictions.shape,
                lambda i: tir.Select(
                    i == targets(), -all_weights() * output_grad(), tir.const(0, predictions.dtype)
                ),
                "pred_grad",
            )

        return te.compute(
            predictions.shape,
            lambda *i: tir.Select(
                i[1] == targets(*i[:1], *i[2:]),
                -all_weights(*i[:1], *i[2:]) * output_grad(*i[:1], *i[2:]),
                tir.const(0, predictions.dtype),
            ),
            "pred_grad",
        )

    def te_nll_loss_backward_no_weight(output_grad, predictions, targets, reduction, ignore_index):
        weight = topi.full(
            (predictions.shape[1] if len(predictions.shape) > 1 else predictions.shape[0],),
            predictions.dtype,
            1.0,
        )
        return te_nll_loss_backward(
            output_grad, predictions, targets, weight, reduction, ignore_index
        )

    if len(call.args) == 3:
        return bb.call_te(
            te_nll_loss_backward_no_weight,
            *call.args,
            reduction=call.attrs.reduction,
            ignore_index=call.attrs.ignore_index,
        )

    return bb.call_te(
        te_nll_loss_backward,
        *call.args,
        reduction=call.attrs.reduction,
        ignore_index=call.attrs.ignore_index,
        primfunc_name_hint="nll_loss_backward",
    )


@register_legalize("relax.grad.max_pool2d_backward")
def _grad_max_pool2d_backward(bb: BlockBuilder, call: Call) -> Expr:
    if not (len(call.attrs.dilation) == 2 and all(i == 1 for i in call.attrs.dilation)):
        logging.info("Dilation is not supported in TOPI pool_grad and is not legalized.")
        return call
    return bb.call_te(
        topi.nn.pool_grad,
        call.args[0],
        call.args[1],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        pool_type="max",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="max_pool2d_backward",
    )


@register_legalize("relax.grad.avg_pool2d_backward")
def _grad_avg_pool2d_backward(bb: BlockBuilder, call: Call) -> Expr:
    if not (len(call.attrs.dilation) == 2 and all(i == 1 for i in call.attrs.dilation)):
        logging.info("Dilation is not supported in TOPI pool_grad and is not legalized.")
        return call
    return bb.call_te(
        topi.nn.pool_grad,
        call.args[0],
        call.args[1],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        pool_type="avg",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="avg_pool2d_backward",
    )


@register_legalize("relax.grad.take_backward")
def _grad_take_backward(bb: BlockBuilder, call: Call) -> Expr:
    axis = call.attrs.axis
    if axis is not None:
        axis = int(axis)

    def te_take_backward(output_grad, x, indices):
        def gen_ir(output_grad_ptr, x_ptr, indices_ptr, out_ptr):
            # pylint: disable=invalid-name
            ib = tir.ir_builder.create()

            output_grad = ib.buffer_ptr(output_grad_ptr)
            indices = ib.buffer_ptr(indices_ptr)
            out = ib.buffer_ptr(out_ptr)

            fused_shape = 1
            for i in x_ptr.shape:
                fused_shape *= i

            with ib.for_range(0, fused_shape) as i:
                out[i] = tir.const(0, dtype=x_ptr.dtype)

            assert len(indices_ptr.shape) == 1  # indices in take must be 1-dim Tensor
            indices_len = indices_ptr.shape[0]

            if axis is not None:
                fused_output_grad_shape_pre = 1
                fused_output_grad_shape_nxt = 1
                for i in range(len(output_grad_ptr.shape)):
                    if i < axis:
                        fused_output_grad_shape_pre *= output_grad_ptr.shape[i]
                    elif i > axis:
                        fused_output_grad_shape_nxt *= output_grad_ptr.shape[i]

                x_axis_len = x_ptr.shape[axis]

                with ib.for_range(
                    0, fused_output_grad_shape_pre * fused_output_grad_shape_nxt, "parallel"
                ) as fused:
                    i = fused // fused_output_grad_shape_nxt
                    j = fused % fused_output_grad_shape_nxt
                    with ib.for_range(0, indices_len, "serial") as l:
                        out[
                            i * fused_output_grad_shape_nxt * x_axis_len
                            + indices[l] * fused_output_grad_shape_nxt
                            + j
                        ] += output_grad[
                            i * fused_output_grad_shape_nxt * indices_len
                            + l * fused_output_grad_shape_nxt
                            + j
                        ]
            else:
                with ib.for_range(0, indices_len, "serial") as l:
                    out[indices[l]] += output_grad[l]

            return ib.get()

        shape = x.shape
        out_buf = tir.decl_buffer(shape, x.dtype, "out_buf")

        return te.extern(
            [shape],
            [output_grad, x, indices],
            lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
            dtype=x.dtype,
            out_buffers=[out_buf],
            name="take_backward",
            tag="take_backward",
        )

    return bb.call_te(
        te_take_backward,
        call.args[0],
        call.args[1],
        call.args[2],
        primfunc_name_hint="take_backward",
    )
