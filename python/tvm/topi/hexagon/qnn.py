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

import tvm
from tvm import te, topi
from ..generic.default import default_schedule as _default_schedule
from ..utils import get_const_tuple
from ..nn.utils import get_pad_tuple
from ..nn.pad import pad
from .. import tag, nn
from ..x86.concat import concatenate


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


def qnn_quantize(data, output_scale, output_zero_point, axis, out_dtype):
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
    return _default_schedule(outs, False)


def qnn_dequantize(data, input_scale, input_zero_point, axis):
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
    return _default_schedule(outs, False)


def qnn_requantize(data, input_scale, input_zp, output_scale, output_zp, axis, out_dtype):
    """Compute for qnn.requantize
    Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))

    TODO: support 'rounding' and 'compute_dtype' arguments.
    """

    def _compute(*indices):
        value = data(*indices)

        # Account scalar and 1D quantization parameters:
        iscale_idx = tvm.tir.indexmod(indices[axis], topi.shape(input_scale)[0])
        iscale = input_scale if len(input_scale.shape) == 0 else input_scale[iscale_idx]

        oscale_idx = tvm.tir.indexmod(indices[axis], topi.shape(output_scale)[0])
        oscale = output_scale if len(output_scale.shape) == 0 else output_scale[oscale_idx]

        sub = te.subtract(value, input_zp)
        mul = te.div(iscale, oscale)
        val = te.add(te.round(te.multiply(mul, sub)), output_zp)

        # clip + cast:
        const_min = tvm.tir.min_value(out_dtype)
        const_max = tvm.tir.max_value(out_dtype)
        return te.max(tvm.te.min(val, const_max), const_min).astype(out_dtype)

    return te.compute(data.shape, _compute)


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
    return _default_schedule(outs, False)


def qnn_add(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Compute for qnn.add
    Q_output = zp_output + round((lhs_scale)/(scale_output) * (lhs_input - lhs_zp_input))
                         + round((rhs_scale)/(scale_output) * (rhs_input - rhs_zp_input))

    TODO: support 'axis' argument.
    """

    assert lhs.dtype == rhs.dtype
    dtype = lhs.dtype

    def _compute(*indices):
        lvalue = lhs(*indices)
        rvalue = rhs(*indices)
        q_lv = te.round(
            te.multiply(te.div(lhs_scale, output_scale), te.subtract(lvalue, lhs_zero_point))
        ).astype("int32")
        q_rv = te.round(
            te.multiply(te.div(rhs_scale, output_scale), te.subtract(rvalue, rhs_zero_point))
        ).astype("int32")
        val = te.add(te.add(q_lv, q_rv), output_zero_point)

        # clip + cast:
        const_min = tvm.tir.min_value(dtype)
        const_max = tvm.tir.max_value(dtype)
        return te.max(tvm.te.min(val, const_max), const_min).astype(dtype)

    return te.compute(lhs.shape, _compute)


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
    return _default_schedule(outs, False)


def requantize_tensor(tensor, i_scale, i_zp, o_scale, o_zp, out_dtype):
    """Requantize tensor"""

    def _compute(*indices):
        value = tensor(*indices)
        mul_value = te.round(
            te.multiply(te.div(i_scale, o_scale), te.subtract(value, i_zp))
        ).astype("int32")
        rq_value = te.add(mul_value, o_zp)

        return clip_cast(rq_value, out_dtype)

    return te.compute(tensor.shape, _compute)


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
        args.append(requantize_tensor(tensor, i_scale, i_zp, o_scale, o_zp, out_dtype))

    # Call x86 implementation of concatenate.
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
    return _default_schedule(outs, False)


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
    axis,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    oshape,
    odtype,
):
    """Compute for qnn.conv2d with NCHW layout"""
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
                n,
                ic,
                oh * height_stride + kh * dilation_h,
                ow * width_stride + kw * dilation_w,
            ].astype("int32")
            * te.subtract(weight[oc, ic, kh, kw], kernel_zero_point).astype("int32"),
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
    return _default_schedule(outs, False)


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
    axis,
    # Conv2d attributes:
    strides,
    padding,
    dilation,
    oshape,
    odtype,
):
    """Compute for qnn.conv2d with NCHW layout"""
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
                n,
                oc,
                oh * height_stride + kh * dilation_h,
                ow * width_stride + kw * dilation_w,
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
    return _default_schedule(outs, False)


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
    axis,
    out_dtype,
):
    """Compute for qnn.dense"""
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    k = te.reduce_axis((0, K), "k")
    # This implementation uses "int32" dense output data type.
    out = te.compute(
        (M, N),
        lambda m, n: te.sum(
            te.subtract(data[m, k], input_zero_point).astype("int32")
            * te.subtract(weight[n, k], kernel_zero_point).astype("int32"),
            axis=k,
        ),
    )

    # Add bias
    if bias is not None:
        out = te.compute(out.shape, lambda n, c: out[n, c] + bias[c])

    # Requantize output of dense
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
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
    return _default_schedule(outs, False)


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
    """Compute for qnn.dense"""

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
    return _default_schedule(outs, False)
