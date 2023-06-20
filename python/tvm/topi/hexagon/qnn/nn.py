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
"""Hexagon QNN operators"""
# pylint: disable=invalid-name

from typing import Union
import numpy as np

import tvm
from tvm import te, topi
from ..utils import (
    saturate,
    is_scalar,
    get_const_int_value,
    get_const_float_value,
    get_fixed_point_value,
)
from ...utils import get_const_tuple
from ...nn.utils import get_pad_tuple
from ...nn.pad import pad
from ... import tag, nn
from ..conv2d import conv2d_NCHWc_int8
from ...transform import concatenate


def clip_cast(val, dtype):
    # clip + cast:
    const_min = tvm.tir.min_value(dtype)
    const_max = tvm.tir.max_value(dtype)
    return te.max(tvm.te.min(val, const_max), const_min).astype(dtype)


def get_qnn_param(param, indices, axis):
    # Account scalar and 1D quantization parameters:
    if len(param.shape) == 0:
        return param

    param_idx = tvm.tir.indexmod(indices[axis], topi.shape(param)[0])
    return param[param_idx]


def subtract_zero_point(tensor: te.Tensor, zero_point: Union[te.Tensor, tvm.tir.IntImm], name: str):
    """
    Subtract zero point from given tensor. If zero point is scalar constant and is equal to 0, then
    it can be optimized and return tensor as it is.
    This new block is marked with 'meta_schedule.inline_rule = disable' attribute to disable inline.
    Otherwise, inline prevents from tensorization and leveraging vrmpy intrinsic
    """
    if is_scalar(zero_point) and get_const_int_value(zero_point) == 0:
        return tensor
    else:
        return te.compute(
            tensor.shape,
            lambda *i: te.subtract(tensor(*i), zero_point).astype(tensor.dtype),
            name=name,
            attrs={"meta_schedule.inline_rule": "disable"},
        )


def default_schedule(outs):
    """Simple default schedule for QNN ops.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    for x in outs:
        fused = s[x].fuse(*x.op.axis)
        outer, inner = s[x].split(fused, factor=128 // np.dtype(x.dtype).itemsize)
        s[x].vectorize(inner)
        s[x].parallel(outer)
    return s


def qnn_quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    """Compute for qnn.quantize

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)
    """

    assert len(output_scale.shape) == 0 or len(output_scale.shape) == 1
    assert len(output_zero_point.shape) == 0 or len(output_zero_point.shape) == 1

    def _compute(*indices):
        value = data(*indices)
        scale = get_qnn_param(output_scale, indices, axis)
        zp = get_qnn_param(output_zero_point, indices, axis)

        val = te.add(te.round(te.div(value, scale)), zp)
        return clip_cast(val, out_dtype)

    return te.compute(data.shape, _compute, tag=tag.ELEMWISE)


def schedule_qnn_quantize(outs):
    """Schedule for qnn.quantize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.quantize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_dequantize(data, input_scale, input_zero_point, axis=-1):
    """Compute for qnn.dequantize

    fp_output = input_scale * (Q_input - input_zero_point)
    """

    def _compute(*indices):
        value = data(*indices)
        scale = get_qnn_param(input_scale, indices, axis)
        zp = get_qnn_param(input_zero_point, indices, axis)

        return te.multiply(scale, te.subtract(value, zp))

    return te.compute(data.shape, _compute, tag=tag.ELEMWISE)


def schedule_qnn_dequantize(outs):
    """Schedule for qnn.dequantize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.dequantize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_requantize(
    data: te.Tensor,
    input_scale: te.Tensor,
    input_zp: te.Tensor,
    output_scale: te.Tensor,
    output_zp: te.Tensor,
    axis=-1,
    out_dtype="int8",
):
    """Compute for qnn.requantize

    If both input and output scales are constant scalars then we convert scale to fixed point value
    and use integer arithmetic only for performance optimization purpose.
    But this is a tradeoff between performance and accuracy, since we use int16 data type to
    represent fixed point values (against QNN lowering approach where we use int32 for that).

    if input and/or output scales are not constant scalars then we use the following formula:
        Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))

    TODO: support 'rounding' and 'compute_dtype' arguments.
    """

    if is_scalar(input_scale) and is_scalar(output_scale):
        iscale = get_const_float_value(input_scale)
        oscale = get_const_float_value(output_scale)
        scale = iscale / oscale
        scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")

        def _compute(*indices):
            value = data(*indices)
            # Subtract input zero point:
            sub = te.subtract(value, input_zp)
            # Fixed point multiply + roundup delta:
            mul = (sub * scale_fixed_point + (1 << (rsh - 1))) >> rsh
            # Add output zero point + clip + cast:
            return saturate(te.add(mul, output_zp), out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize")

    else:

        def _compute(*indices):
            value = data(*indices)
            iscale = get_qnn_param(input_scale, indices, axis)
            oscale = get_qnn_param(output_scale, indices, axis)

            # Subtract input zero point:
            sub = te.subtract(value, input_zp)
            mul = te.div(iscale, oscale)
            val = te.add(te.round(te.multiply(mul, sub)), output_zp)
            # clip + cast:
            return saturate(val, out_dtype).astype(out_dtype)

        return te.compute(data.shape, _compute, name="requantize")


def schedule_qnn_requantize(outs):
    """Schedule for qnn.requantize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.requantize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def compute_qnn_binary_op(
    lhs, rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, output_scale, output_zp, func
):
    """Compute for QNN binary operation

    If rhs/lhs/output scales are constant scalars then we convert scale to fixed point value
    and use integer arithmetic only for performance optimization purpose.
    But this is a tradeoff between performance and accuracy, since we use int16 data type to
    represent fixed point values (against QNN lowering approach where we use int32 for that).

    if rhs/lhs/output scales are not constant scalars then we use the following formula:
        Q_output = output_zp + round((lhs_scale)/(output_scale) * (lhs_input - lhs_zp))
                        _OP_ round((rhs_scale)/(output_scale) * (rhs_input - rhs_zp))
        where _OP_ is add/subtract
    """
    assert lhs.dtype == rhs.dtype
    dtype = lhs.dtype

    def _compute_const(x: te.Tensor, iscale, input_zp):
        return te.round(te.multiply(te.div(iscale, output_scale), te.subtract(x, input_zp))).astype(
            "int32"
        )

    def _compute_tensor(x: te.Tensor, input_scale, input_zp):
        if is_scalar(input_scale) and is_scalar(output_scale):
            iscale = input_scale.op.body[0].value
            oscale = output_scale.op.body[0].value
            scale = iscale / oscale
            scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")
            return te.compute(
                x.shape,
                lambda *i: (te.subtract(x(*i), input_zp) * scale_fixed_point + (1 << (rsh - 1)))
                >> rsh,
            )
        else:
            return te.compute(
                x.shape,
                lambda *i: te.round(
                    te.multiply(te.div(input_scale, output_scale), te.subtract(x(*i), input_zp))
                ).astype("int32"),
            )

    if is_scalar(lhs):
        lhs_tensor = _compute_const(lhs, lhs_scale, lhs_zp)
    else:
        lhs_tensor = _compute_tensor(lhs, lhs_scale, lhs_zp)

    if is_scalar(rhs):
        rhs_tensor = _compute_const(rhs, rhs_scale, rhs_zp)
    else:
        rhs_tensor = _compute_tensor(rhs, rhs_scale, rhs_zp)

    # Binary op with broadcasting
    tensor = func(lhs_tensor, rhs_tensor)

    # Add output zero point and clip+cast.
    def _compute(*indices):
        return saturate(te.add(tensor(*indices), output_zp), dtype).astype(dtype)

    return te.compute(tensor.shape, _compute)


def qnn_add(lhs, rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, output_scale, output_zp):
    """Compute for qnn.add
    TODO: support 'axis' argument.
    """
    return compute_qnn_binary_op(
        lhs, rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, output_scale, output_zp, topi.add
    )


def schedule_qnn_add(outs):
    """Schedule for qnn.add

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.add
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_subtract(lhs, rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, output_scale, output_zp):
    """Compute for qnn.subtract"""

    return compute_qnn_binary_op(
        lhs, rhs, lhs_scale, lhs_zp, rhs_scale, rhs_zp, output_scale, output_zp, topi.subtract
    )


def schedule_qnn_subtract(outs):
    """Schedule for qnn.subtract

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.add
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_mul(
    lhs: te.Tensor,
    rhs: te.Tensor,
    lhs_scale: te.Tensor,
    lhs_zp: te.Tensor,
    rhs_scale: te.Tensor,
    rhs_zp: te.Tensor,
    output_scale: te.Tensor,
    output_zp: te.Tensor,
):
    """Compute for qnn.mul

    mul = (lhs_input - lhs_zp) * (rhs_input - rhs_zp)
    Q_output = requantize(mul, lhs_scale * rhs_scale, 0, output_scale, output_zp)
    """
    assert lhs.dtype == rhs.dtype
    odtype = lhs.dtype

    def _compute_tensor(tensor, zero_point):
        if is_scalar(tensor):
            return tensor - zero_point
        else:
            return te.compute(tensor.shape, lambda *i: te.subtract(tensor(*i), zero_point))

    lhs_tensor = _compute_tensor(lhs, lhs_zp)
    rhs_tensor = _compute_tensor(rhs, rhs_zp)

    # Multiply with broadcasting.
    mul = topi.multiply(lhs_tensor, rhs_tensor)

    if is_scalar(lhs_scale) and is_scalar(rhs_scale):
        assert isinstance(lhs_scale, te.Tensor)
        assert isinstance(rhs_scale, te.Tensor)
        iscale = lhs_scale.op.body[0] * rhs_scale.op.body[0]
    else:
        iscale = lhs_scale * rhs_scale

    return qnn_requantize(mul, iscale, tvm.tir.const(0), output_scale, output_zp, out_dtype=odtype)


def schedule_qnn_mul(outs):
    """Schedule for qnn.mul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.add
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_tanh(data, input_scale, input_zp, output_scale, output_zp):
    """Compute for qnn.tanh

    Q_output = quantize(tanh(dequantize(data)))
    """
    dq_tensor = qnn_dequantize(data, input_scale, input_zp)
    tanh = te.compute(dq_tensor.shape, lambda *i: te.tanh(dq_tensor(*i)))
    return qnn_quantize(tanh, output_scale, output_zp, out_dtype=data.dtype)


def schedule_qnn_tanh(outs):
    """Schedule for qnn.tanh

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.add
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_concatenate(data, axis, out_dtype):
    """Compute for qnn.concatenate

    Parameters
    ----------
    data: Array of Tensor
          The computation graph description of qnn.concatenate
          in the format of an array of tensors.

    axis: int
          The axis along which the tensors are concatenated.

    out_dtype: string
          Data type of output tensor

    Returns
    -------
    out: Tensor
        The computation for the op.
    """

    # Get output quantization parameters.
    o_scale = data[-2]
    o_zp = data[-1]

    # Initially qnn.concatenate had 3 tuples: (1) tuple with input tensors, (2) tuple with input
    # scales and (3) tuple with input zero points.
    # Last 2 elements in data represent output scale and zero point.
    num_of_tuples = 3
    assert ((len(data) - 2) % num_of_tuples) == 0
    args_num = (len(data) - 2) // num_of_tuples

    args = []
    for i in range(args_num):
        # Get next tensor and its quantization parameters.
        tensor = data[i]
        i_scale = data[i + args_num]
        i_zp = data[i + args_num * 2]

        # Requantize tensors and add them to the list.
        args.append(qnn_requantize(tensor, i_scale, i_zp, o_scale, o_zp, out_dtype=out_dtype))

    # Call generic implementation of concatenate.
    return concatenate(args, axis)


def schedule_qnn_concatenate(outs):
    """Schedule for qnn.concatenate

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.add
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_conv2d(  # Conv2d inputs
    data,
    weight,
    # Conv2d quantization params:
    input_zero_point,
    kernel_zero_point,
    _input_scale,
    _kernel_scale,
    # bias
    bias,
    # Requantization params:
    rq_input_scale,
    rq_input_zero_point,
    rq_output_scale,
    rq_output_zero_point,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    oshape,
    odtype,
):
    """Compute for qnn.conv2d with NCHW layout.

    Output data type should be specified through the 'odtype' parameter. qnn.conv2d leverages int32
    type to store intermediate results. If 'odtype' differs from int32, you need to specify
    requantization parameters.
    """
    in_channel = data.shape[1]  # NCHW layout
    kernel_height = weight.shape[2]  # OIHW layout
    kernel_width = weight.shape[3]  # OIHW layout

    height_stride, width_stride = strides
    dilation_h, dilation_w = dilation

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kernel_h, dilated_kernel_w)
    )

    # Subtract zero point from weights. axis=0 in get_qnn_param means 'O' dimension in "OIHW"
    # weights layout.
    weight = te.compute(
        weight.shape,
        lambda *indices: te.subtract(
            weight(*indices), get_qnn_param(kernel_zero_point, indices, axis=0)
        ),
    )

    # Subtract zero point from input and then do padding with 0 value
    data = te.compute(data.shape, lambda *indices: te.subtract(data(*indices), input_zero_point))

    # DOPAD
    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, 0, pad_top, pad_left)
        pad_after = (0, 0, pad_down, pad_right)
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    out = te.compute(
        oshape,
        lambda n, oc, oh, ow: te.sum(
            data_pad[
                n, ic, oh * height_stride + kh * dilation_h, ow * width_stride + kw * dilation_w
            ].astype("int32")
            * weight[oc, ic, kh, kw].astype("int32"),
            axis=[ic, kh, kw],
        ),
    )

    # Add bias
    if bias is not None:
        assert len(out.shape) == len(bias.shape)
        assert bias.shape[2] == 1 and bias.shape[3] == 1
        out = te.compute(out.shape, lambda n, c, h, w: out[n, c, h, w] + bias[n, c, 0, 0])

    # Requantize output of convolution
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert len(rq_input_scale.shape) == 0 or len(rq_input_scale.shape) == 1
        assert len(rq_output_scale.shape) == 0 or len(rq_output_scale.shape) == 1
        axis = -1
        if len(rq_input_scale.shape) == 1 or len(rq_output_scale.shape) == 1:
            axis = 1  # Axis param should correspond to 'C' dimension.

        return qnn_requantize(
            out,
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
            axis,
            odtype,
        )

    return out


def schedule_qnn_conv2d(outs):
    """Schedule for qnn.conv2d

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.conv2d
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_conv2d_NCHWc_int8(  # Conv2d inputs
    data,
    weight,
    # Conv2d quantization params:
    input_zero_point,
    kernel_zero_point,
    _input_scale,
    _kernel_scale,
    # bias
    bias,
    # Requantization params:
    rq_input_scale,
    rq_input_zero_point,
    rq_output_scale,
    rq_output_zero_point,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    _oshape,
    odtype,
):
    """Compute for qnn.conv2d with NCHWc layout."""

    # Subtract zero point from input and weights.
    weight = subtract_zero_point(weight, kernel_zero_point, "weight_zp")
    data = subtract_zero_point(data, input_zero_point, "data_zp")

    strides = get_const_tuple(strides)
    padding = get_const_tuple(padding)
    dilation = get_const_tuple(dilation)
    out = conv2d_NCHWc_int8(data, weight, strides, padding, dilation, "NCHW32c", "NCHW32c")

    # Add bias
    if bias is not None:
        assert len(out.shape) == len(bias.shape)
        assert bias.shape[2] == 1 and bias.shape[3] == 1
        out = te.compute(
            out.shape,
            lambda n, c, h, w, ci: out[n, c, h, w, ci] + bias[n, c, 0, 0, ci],
            name="bias_add",
        )

    # Requantize output of convolution
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert len(rq_input_scale.shape) == 0 or len(rq_input_scale.shape) == 1
        assert len(rq_output_scale.shape) == 0 or len(rq_output_scale.shape) == 1
        axis = -1
        if len(rq_input_scale.shape) == 1 or len(rq_output_scale.shape) == 1:
            axis = 1  # Axis param should correspond to 'C' dimension.

        return qnn_requantize(
            out,
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
            axis,
            odtype,
        )

    return out


def schedule_qnn_conv2d_NCHWc_int8(outs):
    """Schedule for qnn.conv2d with NCHWc layout."""

    return default_schedule(outs)


def qnn_depthwise_conv2d(  # Conv2d inputs
    data,
    weight,
    # Conv2d quantization params:
    input_zero_point,
    kernel_zero_point,
    _input_scale,
    _kernel_scale,
    # bias
    bias,
    # Requantization params:
    rq_input_scale,
    rq_input_zero_point,
    rq_output_scale,
    rq_output_zero_point,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    oshape,
    odtype,
):
    """Compute for qnn.conv2d with NCHW layout

    Output data type should be specified through the 'odtype' parameter. qdepthwise nn.conv2d
    leverages int32 type to store intermediate results. If 'odtype' differs from int32, you need to
    specify requantization parameters.
    """
    kernel_height = weight.shape[2]  # OIHW layout
    kernel_width = weight.shape[3]  # OIHW layout

    height_stride, width_stride = strides
    dilation_h, dilation_w = dilation

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kernel_h, dilated_kernel_w)
    )

    # Subtract zero point from input and then do padding with 0 value
    data = te.compute(data.shape, lambda *indices: te.subtract(data(*indices), input_zero_point))

    # DOPAD
    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, 0, pad_top, pad_left)
        pad_after = (0, 0, pad_down, pad_right)
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    out = te.compute(
        oshape,
        lambda n, oc, oh, ow: te.sum(
            data_pad[
                n, oc, oh * height_stride + kh * dilation_h, ow * width_stride + kw * dilation_w
            ].astype("int32")
            * te.subtract(weight[oc, 0, kh, kw], kernel_zero_point).astype("int32"),
            axis=[kh, kw],
        ),
    )

    # Add bias
    if bias is not None:
        assert len(out.shape) == len(bias.shape)
        assert bias.shape[2] == 1 and bias.shape[3] == 1
        out = te.compute(out.shape, lambda n, c, h, w: out[n, c, h, w] + bias[n, c, 0, 0])

    # Requantize output of convolution
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert len(rq_input_scale.shape) == 0 or len(rq_input_scale.shape) == 1
        assert len(rq_output_scale.shape) == 0 or len(rq_output_scale.shape) == 1
        axis = -1
        if len(rq_input_scale.shape) == 1 or len(rq_output_scale.shape) == 1:
            axis = 1  # Axis param should correspond to 'C' dimension.

        return qnn_requantize(
            out,
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
            axis,
            odtype,
        )

    return out


def schedule_qnn_depthwise_conv2d(outs):
    """Schedule for depthwise qnn.conv2d

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.conv2d
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_dense(
    data,
    weight,
    # Dense quantization params:
    input_zero_point,
    kernel_zero_point,
    _input_scale,
    _kernel_scale,
    # bias
    bias,
    # Requantization params:
    rq_input_scale,
    rq_input_zero_point,
    rq_output_scale,
    rq_output_zero_point,
    out_dtype,
):
    """Compute for qnn.dense

    Output data type should be specified through the 'odtype' parameter. qnn.dense leverages int32
    type to store intermediate results. If 'odtype' differs from int32, you need to specify
    requantization parameters.
    """
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    k = te.reduce_axis((0, K), "k")
    # This implementation uses "int32" dense output data type.
    # axis=0 in get_qnn_param mean 'N' dimension in "NK" weights layout.
    out = te.compute(
        (M, N),
        lambda m, n: te.sum(
            te.subtract(data[m, k], input_zero_point).astype("int32")
            * te.subtract(weight[n, k], get_qnn_param(kernel_zero_point, (n, k), axis=0)).astype(
                "int32"
            ),
            axis=k,
        ),
    )

    # Add bias
    if bias is not None:
        out = te.compute(out.shape, lambda n, c: out[n, c] + bias[0, c])

    # Requantize output of dense
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert len(rq_input_scale.shape) == 0 or len(rq_input_scale.shape) == 1
        assert len(rq_output_scale.shape) == 0 or len(rq_output_scale.shape) == 1
        axis = -1
        if len(rq_input_scale.shape) == 1 or len(rq_output_scale.shape) == 1:
            axis = 1  # Axis param should correspond to 'N' dimension.

        return qnn_requantize(
            out,
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
            axis,
            out_dtype,
        )

    return out


def schedule_qnn_dense(outs):
    """Schedule for qnn.dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_dense_pack_vrmpy(
    data: te.Tensor,
    weight: te.Tensor,
    # Dense quantization params:
    input_zero_point: te.Tensor,
    kernel_zero_point: te.Tensor,
    _input_scale: te.Tensor,
    _kernel_scale: te.Tensor,
    # bias
    bias: te.Tensor,
    # Requantization params:
    rq_input_scale: te.Tensor,
    rq_input_zero_point: te.Tensor,
    rq_output_scale: te.Tensor,
    rq_output_zero_point: te.Tensor,
    out_dtype: str,
):
    """Compute for qnn.contrib_dense_pack

    Output data type should be specified through the 'odtype' parameter. qnn.dense leverages int32
    type to store intermediate results. If 'odtype' differs from int32, you need to specify
    requantization parameters.
    """
    # Subtract zero point from input and weights.
    weight = subtract_zero_point(weight, kernel_zero_point, "weight_zp")
    data = subtract_zero_point(data, input_zero_point, "data_zp")

    # Required for vrmpy intrinsic
    assert "int8" in weight.dtype and "int8" in data.dtype

    M, K = get_const_tuple(data.shape)
    N_O, _, N_I, _ = get_const_tuple(weight.shape)
    k = te.reduce_axis((0, K), "k")
    out = te.compute(
        (M, N_O * N_I),
        lambda m, n: te.sum(
            data[m, k].astype("int32")
            * weight[
                tvm.tir.indexdiv(n, 32),
                tvm.tir.indexdiv(k, 4),
                tvm.tir.indexmod(n, 32),
                tvm.tir.indexmod(k, 4),
            ].astype("int32"),
            axis=k,
        ),
        name="qnn_dense_pack",
    )

    # Add bias
    if bias is not None:
        assert bias.ndim == 2
        out = te.compute(out.shape, lambda n, c: out[n, c] + bias[0, c])

    # Requantize output of qnn.contrib_dense_pack
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert rq_input_scale.ndim == 0 or rq_input_scale.ndim == 1
        assert rq_output_scale.ndim == 0 or rq_output_scale.ndim == 1
        axis = -1
        if rq_input_scale.ndim == 1 or rq_output_scale.ndim == 1:
            axis = 1  # Axis param should correspond to 'C' dimension.

        return qnn_requantize(
            out,
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
            axis,
            out_dtype,
        )

    return out


def schedule_qnn_dense_pack_vrmpy(outs):
    """Schedule for qnn.contrib_dense_pack

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)


def qnn_batch_matmul(
    tensor_a,
    tensor_b,
    # batch_matmul quantization params:
    a_zero_point,
    b_zero_point,
    _a_scale,
    _b_scale,
    # Attributes
    transpose_a,
    transpose_b,
    out_dtype,
):
    """Compute for qnn.batch_matmul"""

    # Preprocess tensor_a: subtract zp
    a_sub_zp = te.compute(
        tensor_a.shape, lambda *indices: te.subtract(tensor_a(*indices), a_zero_point)
    )
    # Preprocess tensor_b: subtract zp
    b_sub_zp = te.compute(
        tensor_b.shape, lambda *indices: te.subtract(tensor_b(*indices), b_zero_point)
    )

    return nn.batch_matmul(a_sub_zp, b_sub_zp, None, out_dtype, transpose_a, transpose_b)


def schedule_qnn_batch_matmul(outs):
    """Schedule for qnn.batch_matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return default_schedule(outs)
