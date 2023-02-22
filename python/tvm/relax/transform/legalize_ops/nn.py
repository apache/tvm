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
"""Default legalization function for neural network operators."""
import logging

from tvm import topi, tir, te
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize, _call_topi_without_attr


@register_legalize("relax.nn.conv2d")
def _nn_conv2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if len(call.attrs.data_layout) != 4 or len(call.attrs.kernel_layout) != 4:
        logging.info(
            "Conv2D where data layout or kernel layout have channel chunk "
            "cannot be legalized by TOPI at this moment."
        )
        return call
    if call.attrs.groups != 1:
        data_layout = tir.layout(call.attrs.data_layout)
        kernel_layout = tir.layout(call.attrs.kernel_layout)
        ic = call.args[0].struct_info.shape.values[data_layout.index_of("C")]
        oc = call.args[1].struct_info.shape.values[kernel_layout.index_of("O")]
        if not isinstance(ic, tir.IntImm) or not isinstance(oc, tir.IntImm):
            logging.info(
                "Conv2D where number of groups is more than one and input or output "
                "channel size is symbolic cannot be legalized by TOPI at this moment."
            )
            return call

    return bb.call_te(
        topi.nn.conv,
        inp=call.args[0],
        filt=call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        groups=call.attrs.groups,
        data_layout=call.attrs.data_layout,
        kernel_layout=call.attrs.kernel_layout,
        out_dtype=call.attrs.out_dtype if call.attrs.out_dtype != "" else None,
        primfunc_name_hint="conv2d",
    )


@register_legalize("relax.nn.max_pool2d")
def _nn_max_pool2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI max_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool2d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="max",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="max_pool2d",
    )


@register_legalize("relax.nn.adaptive_avg_pool2d")
def _nn_adaptive_avg_pool2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI adaptive_avg_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    def te_adaptive_avg_pool2d(data, output_size, layout_str):
        if output_size is None:
            layout = tir.layout(layout_str)
            idx_H = layout.index_of("H")
            idx_W = layout.index_of("W")
            assert idx_H != -1 and idx_W != -1
            output_size = (data.shape[idx_H], data.shape[idx_W])

        return topi.nn.adaptive_pool(data, output_size, "avg", layout_str)

    return bb.call_te(
        te_adaptive_avg_pool2d,
        call.args[0],
        call.attrs.output_size,
        call.attrs.layout,
        primfunc_name_hint="adaptive_avg_pool2d",
    )


register_legalize("relax.nn.relu", _call_topi_without_attr(topi.nn.relu))


@register_legalize("relax.nn.gelu")
def _nn_gelu(bb: BlockBuilder, call: Call) -> Expr:
    def te_gelu(x: te.Tensor):
        dtype = x.dtype
        return x * (
            tir.const(0.5, dtype)
            + topi.erf(x * tir.const(0.5**0.5, dtype)) * tir.const(0.5, dtype)
        )

    return bb.call_te(te_gelu, call.args[0], primfunc_name_hint="gelu")


@register_legalize("relax.nn.silu")
def _nn_silu(bb: BlockBuilder, call: Call) -> Expr:
    def te_silu(x: te.Tensor):
        return topi.multiply(x, topi.sigmoid(x))

    return bb.call_te(te_silu, call.args[0], primfunc_name_hint="silu")


@register_legalize("relax.nn.softmax")
def _nn_softmax(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.nn.softmax, call.args[0], call.attrs.axis)


@register_legalize("relax.nn.log_softmax")
def _nn_log_softmax(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.nn.log_softmax, call.args[0], call.attrs.axis)


@register_legalize("relax.nn.cross_entropy_with_logits")
def _nn_cross_entropy_with_logits(bb: BlockBuilder, call: Call):
    def te_cross_entropy_with_logits(x, y):
        if len(x.shape) > 1:
            return -topi.sum(x * y) / x.shape[0]
        return -topi.sum(x * y)

    return bb.call_te(
        te_cross_entropy_with_logits,
        call.args[0],
        call.args[1],
        primfunc_name_hint="cross_entropy_with_logits",
    )


@register_legalize("relax.nn.batch_norm")
def _nn_batch_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.batch_norm,
        data=call.args[0],
        gamma=call.args[1],
        beta=call.args[2],
        moving_mean=call.args[3],
        moving_var=call.args[4],
        axis=call.attrs.axis,
        epsilon=call.attrs.epsilon,
        center=call.attrs.center,
        scale=call.attrs.scale,
    )


@register_legalize("relax.nn.layer_norm")
def _nn_layer_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.layer_norm,
        call.args[0],
        call.args[1],
        call.args[2],
        axis=call.attrs.axes,
        epsilon=call.attrs.epsilon,
    )


@register_legalize("relax.nn.dropout")
def _nn_dropout(bb: BlockBuilder, call: Call) -> Expr:
    logging.info("Dropout is handled by frontend translator at this moment and is not legalized.")
    return call
