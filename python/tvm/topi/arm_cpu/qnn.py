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
import tvm
from tvm import te, topi
from ..utils import get_const_tuple
from ..nn.utils import get_pad_tuple
from ..nn.pad import pad
from .. import tag, nn

def get_qnn_param(param, indices, axis):
    # Account scalar and 1D quantization parameters:
    if len(param.shape) == 0:
        return param

    param_idx = tvm.tir.indexmod(indices[axis], topi.shape(param)[0])
    return param[param_idx]


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
    return s


def qnn_requantize(data, input_scale, input_zp, output_scale, output_zp, axis, out_dtype):
    """Compute for qnn.requantize

    Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))

    TODO: support 'rounding' and 'compute_dtype' arguments.
    """

    def _compute(*indices):
        value = data(*indices)

        iscale = get_qnn_param(input_scale, indices, axis)
        oscale = get_qnn_param(output_scale, indices, axis)
        izp = get_qnn_param(input_zp, indices, axis)


        sub = te.subtract(value, izp)
        mul = te.div(iscale, oscale)
        val = te.add(te.round(te.multiply(mul, sub)), output_zp)

        # clip + cast:
        const_min = tvm.tir.min_value(out_dtype)
        const_max = tvm.tir.max_value(out_dtype)
        return te.max(tvm.te.min(val, const_max), const_min).astype(out_dtype)

    return te.compute(data.shape, _compute)


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
    """Compute for qnn.conv2d with NHWC layout. Note that this is a DIFFERENT layout from the
    Hexagon variant, because they have special instructions Cortex-M doesn't have. We also expect
    the kernel to have OHWI layout.
    """
    in_channel = data.shape[3]  # NHWC layout
    kernel_height = weight.shape[1]  # OHWI layout
    kernel_width = weight.shape[2]  # OHWI layout

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
    ic = te.reduce_axis((0, in_channel), name="ic")

    # axis=0 in get_qnn_param means 'O' dimension in "OIHW" weights layout.
    out = te.compute(
        oshape,
        lambda n, oh, ow, oc: te.sum(
            data_pad[
                n,
                oh * height_stride + kh * dilation_h,
                ow * width_stride + kw * dilation_w,
                ic,
            ].astype("int32")
            * te.subtract(
                weight[oc, kh, kw, ic], get_qnn_param(kernel_zero_point, (oc, kh, kw, ic), axis=0)
            ).astype("int32"),
            axis=[kh, kw, ic],
        ),
    )

    # Add bias
    if bias is not None:
        assert len(out.shape) == len(bias.shape)
        assert bias.shape[1] == bias.shape[2] == 1
        out = te.compute(out.shape, lambda n, h, w, c: out[n, h, w, c] + bias[n, 0, 0, c])

    # Requantize output of convolution
    # Q_output = zp_output + round((scale_input)/(scale_output) * (Q_input - zp_input))
    if rq_input_scale is not None and rq_output_scale is not None:
        # Now supported only scalar and 1D quantization parameters
        assert len(rq_input_scale.shape) == 0 or len(rq_input_scale.shape) == 1
        assert len(rq_output_scale.shape) == 0 or len(rq_output_scale.shape) == 1
        axis = -1
        if len(rq_input_scale.shape) == 1 or len(rq_output_scale.shape) == 1:
            axis = 3  # Axis param should correspond to 'C' dimension.

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