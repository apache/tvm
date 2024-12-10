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
import math
from typing import Optional

from tvm import te, tir, topi

from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import _call_topi_without_attr, register_legalize


@register_legalize("relax.nn.conv1d")
def _nn_conv1d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv1d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if len(call.attrs.data_layout) != 3 or len(call.attrs.kernel_layout) != 3:
        logging.info(
            "Conv1D where data layout or kernel layout have channel chunk "
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
                "Conv1D where number of groups is more than one and input or output "
                "channel size is symbolic cannot be legalized by TOPI at this moment."
            )
            return call

    return bb.call_te(
        topi.nn.conv1d,
        data=call.args[0],
        kernel=call.args[1],
        strides=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        groups=call.attrs.groups,
        data_layout=call.attrs.data_layout,
        kernel_layout=call.attrs.kernel_layout,
        out_dtype=call.attrs.out_dtype if call.attrs.out_dtype != "" else None,
        primfunc_name_hint="conv1d",
    )


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


@register_legalize("relax.nn.conv3d")
def _nn_conv3d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv3d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if len(call.attrs.data_layout) != 5 or len(call.attrs.kernel_layout) != 5:
        logging.info(
            "Conv3D where data layout or kernel layout have channel chunk "
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
                "Conv3D where number of groups is more than one and input or output "
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
        primfunc_name_hint="conv3d",
    )


@register_legalize("relax.nn.conv1d_transpose")
def _nn_conv1d_transpose(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv1d_transpose does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if call.attrs.data_layout != "NCW" or call.attrs.kernel_layout != "IOW":
        logging.info(
            "TOPI conv1d_transpose does not support input layout other than NCW, "
            "and kernel layout other than IOW, so cannot be legalized by TOPI"
        )
        return call
    dilation = call.attrs.dilation
    if len(dilation) != 1 or dilation[0] != 1:
        logging.info(
            "TOPI conv1d_transpose does not support dilations other than 1, "
            "and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.group_conv1d_transpose_ncw,
        call.args[0],
        call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        out_dtype=call.struct_info.dtype,
        output_padding=call.attrs.output_padding,
        groups=call.attrs.groups,
        primfunc_name_hint="conv1d_transpose",
    )


@register_legalize("relax.nn.conv2d_transpose")
def _nn_conv2d_transpose(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv2d_transpose does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call
    if call.attrs.data_layout != "NCHW" or call.attrs.kernel_layout != "IOHW":
        logging.info(
            "TOPI conv2d_transpose does not support input layout other than NCHW, "
            "and kernel layout other than IOHW, so cannot be legalized by TOPI"
        )
        return call
    dilation = call.attrs.dilation
    if len(dilation) != 2 or dilation[0] != 1 or dilation[1] != 1:
        logging.info(
            "TOPI conv2d_transpose does not support dilations other than 1, "
            "and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.group_conv2d_transpose_nchw,
        call.args[0],
        call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        out_dtype=call.struct_info.dtype,
        output_padding=call.attrs.output_padding,
        groups=call.attrs.groups,
        primfunc_name_hint="conv2d_transpose",
    )


@register_legalize("relax.nn.pad")
def _nn_pad(bb: BlockBuilder, call: Call) -> Expr:
    # Unpack pad_width into two separate lists for topi.
    pad_widths = call.attrs.pad_width
    pad_before = pad_widths[::2]
    pad_after = pad_widths[1::2]
    return bb.call_te(
        topi.nn.pad,
        call.args[0],
        pad_before=pad_before,
        pad_after=pad_after,
        pad_value=float(call.args[1].data.numpy()),
        primfunc_name_hint="pad",
    )


@register_legalize("relax.nn.max_pool1d")
def _nn_max_pool1d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI max_pool1d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool1d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="max",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="max_pool1d",
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


@register_legalize("relax.nn.max_pool3d")
def _nn_max_pool3d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI max_pool3d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool3d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="max",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        primfunc_name_hint="max_pool3d",
    )


@register_legalize("relax.nn.avg_pool1d")
def _nn_avg_pool1d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI avg_pool1d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool1d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="avg",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        count_include_pad=call.attrs.count_include_pad,
        primfunc_name_hint="avg_pool1d",
    )


@register_legalize("relax.nn.avg_pool2d")
def _nn_avg_pool2d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI avg_pool2d does not support different input-output "
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
        pool_type="avg",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        count_include_pad=call.attrs.count_include_pad,
        primfunc_name_hint="avg_pool2d",
    )


@register_legalize("relax.nn.avg_pool3d")
def _nn_avg_pool3d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI avg_pool3d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool3d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="avg",
        ceil_mode=call.attrs.ceil_mode,
        layout=call.attrs.layout,
        count_include_pad=call.attrs.count_include_pad,
        primfunc_name_hint="avg_pool3d",
    )


@register_legalize("relax.nn.adaptive_avg_pool1d")
def _nn_adaptive_avg_pool1d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI adaptive_avg_pool1d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    def te_adaptive_avg_pool1d(data, output_size, layout_str):
        if output_size is None:
            layout = tir.layout(layout_str)
            idx_W = layout.index_of("W")
            assert idx_W != -1
            output_size = data.shape[idx_W]

        return topi.nn.adaptive_pool1d(data, output_size, "avg", layout_str)

    return bb.call_te(
        te_adaptive_avg_pool1d,
        call.args[0],
        call.attrs.output_size,
        call.attrs.layout,
        primfunc_name_hint="adaptive_avg_pool1d",
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


@register_legalize("relax.nn.adaptive_avg_pool3d")
def _nn_adaptive_avg_pool3d(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI adaptive_avg_pool3d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    def te_adaptive_avg_pool3d(data, output_size, layout_str):
        if output_size is None:
            layout = tir.layout(layout_str)
            idx_D = layout.index_of("D")
            idx_H = layout.index_of("H")
            idx_W = layout.index_of("W")
            assert idx_D != -1 and idx_H != -1 and idx_W != -1
            output_size = (data.shape[idx_D], data.shape[idx_H], data.shape[idx_W])

        return topi.nn.adaptive_pool3d(data, output_size, "avg", layout_str)

    return bb.call_te(
        te_adaptive_avg_pool3d,
        call.args[0],
        call.attrs.output_size,
        call.attrs.layout,
        primfunc_name_hint="adaptive_avg_pool3d",
    )


register_legalize("relax.nn.relu", _call_topi_without_attr(topi.nn.relu))


@register_legalize("relax.nn.leakyrelu")
def _nn_leakyrelu(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(topi.nn.leaky_relu, call.args[0], call.attrs.alpha)


@register_legalize("relax.nn.gelu")
def _nn_gelu(bb: BlockBuilder, call: Call) -> Expr:
    def te_gelu(x: te.Tensor):
        dtype = x.dtype
        erf_inp = x * tir.const(0.5**0.5, dtype)

        if dtype == "float16":
            erf = topi.math.cast(topi.erf(topi.math.cast(erf_inp, "float32")), "float16")
        else:
            erf = topi.erf(erf_inp)

        return x * (tir.const(0.5, dtype) + erf * tir.const(0.5, dtype))

    return bb.call_te(te_gelu, call.args[0], primfunc_name_hint="gelu")


@register_legalize("relax.nn.gelu_tanh")
def _nn_gelu_tanh(bb: BlockBuilder, call: Call) -> Expr:
    def te_gelu_tanh(x: te.Tensor):
        dtype = x.dtype
        return (
            tir.const(0.5, dtype)
            * x
            * (
                tir.const(1.0, dtype)
                + topi.tanh(
                    tir.const(math.sqrt(2.0 / math.pi), dtype)
                    * x
                    * (1 + tir.const(0.044715, dtype) * x * x)
                )
            )
        )

    return bb.call_te(te_gelu_tanh, call.args[0], primfunc_name_hint="gelu_tanh")


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
        # By default relax batch_norm is training mode.
        # To transform it to inference mode, use DecomposeOpsForInference.
        training=True,
        momentum=call.attrs.momentum,
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


@register_legalize("relax.nn.group_norm")
def _nn_group_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.group_norm,
        call.args[0],
        call.args[1],
        call.args[2],
        call.attrs.num_groups,
        call.attrs.channel_axis,
        call.attrs.axes,
        call.attrs.epsilon,
    )


@register_legalize("relax.nn.rms_norm")
def _nn_rms_norm(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.nn.rms_norm,
        call.args[0],
        call.args[1],
        axis=call.attrs.axes,
        epsilon=call.attrs.epsilon,
    )


@register_legalize("relax.nn.dropout")
def _nn_dropout(bb: BlockBuilder, call: Call) -> Expr:
    logging.info("Dropout is handled by frontend translator at this moment and is not legalized.")
    return call


def _te_attention(
    q: te.Tensor,
    k: te.Tensor,
    v: te.Tensor,
    bias: te.Tensor,
    scale: tir.FloatImm,
    causal_mask: Optional[str],
) -> te.Tensor:
    batch_size, seq_len, num_head, head_dim = q.shape
    _, seq_len_kv, _, head_dim_v = v.shape
    q = topi.transpose(q, [0, 2, 1, 3])
    k = topi.transpose(k, [0, 2, 1, 3])
    v = topi.transpose(v, [0, 2, 1, 3])
    q = topi.reshape(q, [batch_size * num_head, seq_len, head_dim])
    k = topi.reshape(k, [batch_size * num_head, seq_len_kv, head_dim])
    v = topi.reshape(v, [batch_size * num_head, seq_len_kv, head_dim_v])
    p = topi.nn.batch_matmul(q, k)
    if scale is not None:
        p = topi.multiply(p, scale)
    else:
        p = topi.divide(p, tir.sqrt(tir.Cast(p.dtype, head_dim)))
    if bias is not None:
        p = topi.reshape(p, [batch_size, num_head, seq_len, seq_len_kv])
        p = topi.add(p, bias)
        p = topi.reshape(p, [batch_size * num_head, seq_len, seq_len_kv])
    if causal_mask is None:
        s = topi.nn.softmax(p)
    else:
        if causal_mask == "TopLeft":
            offset = tir.IntImm("int32", 0)
        elif causal_mask == "BottomRight":
            offset = tir.abs(seq_len - seq_len_kv).astype("int32")
        else:
            raise NotImplementedError()
        p_masked = topi.trilu(p, k=offset, upper=False)
        p_masked_exp = topi.trilu(
            topi.exp(p_masked - topi.max(p_masked, axis=-1, keepdims=True)), k=offset, upper=False
        )
        p_masked_sum = topi.sum(p_masked_exp, axis=-1, keepdims=True)
        s = topi.divide(p_masked_exp, p_masked_sum)
    o = topi.nn.batch_matmul(s, v, transpose_b=False)
    o = topi.reshape(o, [batch_size, num_head, seq_len, head_dim_v])
    return topi.transpose(o, [0, 2, 1, 3])


@register_legalize("relax.nn.attention")
def _nn_attention(bb: BlockBuilder, call: Call) -> Expr:
    assert (
        call.attrs.window_size is None
    ), "Legalization for sliding-window attention is not supported yet."
    return bb.call_te(
        _te_attention,
        call.args[0],
        call.args[1],
        call.args[2],
        None,
        call.attrs.scale,
        call.attrs.causal_mask,
        primfunc_name_hint="attention",
    )


@register_legalize("relax.nn.attention_bias")
def _nn_attention_bias(bb: BlockBuilder, call: Call) -> Expr:
    assert (
        call.attrs.window_size is None
    ), "Legalization for sliding-window attention is not supported yet."
    return bb.call_te(
        _te_attention,
        call.args[0],
        call.args[1],
        call.args[2],
        call.args[3],
        call.attrs.scale,
        call.attrs.causal_mask,
        primfunc_name_hint="attention_bias",
    )


@register_legalize("relax.nn.attention_var_len")
def _nn_attention_var_len(bb: BlockBuilder, call: Call) -> Expr:
    raise RuntimeError("Legalization of attention_var_len op is not supported yet.")


@register_legalize("relax.nn.nll_loss")
def _nn_nll_loss(bb: BlockBuilder, call: Call) -> Expr:
    def nll_loss_without_weight(predictions, targets, reduction, ignore_index):
        weight = topi.full(
            (predictions.shape[1] if len(predictions.shape) > 1 else predictions.shape[0],),
            predictions.dtype,
            1.0,
        )
        return topi.nn.nll_loss(predictions, targets, weight, reduction, ignore_index)

    if len(call.args) == 2:
        return bb.call_te(
            nll_loss_without_weight,
            call.args[0],
            call.args[1],
            reduction=call.attrs.reduction,
            ignore_index=call.attrs.ignore_index,
        )

    return bb.call_te(
        topi.nn.nll_loss,
        call.args[0],
        call.args[1],
        call.args[2],
        reduction=call.attrs.reduction,
        ignore_index=call.attrs.ignore_index,
    )
