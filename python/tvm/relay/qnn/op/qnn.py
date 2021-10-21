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
# pylint: disable=invalid-name
"""QNN dialect operators."""

from __future__ import absolute_import as _abs

from tvm import relay
from tvm.relay.expr import Tuple, TupleWrapper
from tvm.relay.op.nn.utils import get_pad_tuple2d
from tvm.topi.nn.qnn import SQNN_DTYPE_TO_CODE

from ... import op as reg
from ...op import OpPattern
from . import _make


def requantize(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    rounding="UPWARD",
    out_dtype="int8",
):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: tvm.relay.Expr
        The quantization scale for the input tensor.

    input_zero_point: tvm.relay.Expr
        The zero point of the input tensor.

    output_scale: tvm.relay.Expr
        The quantization scale for the output tensor.

    output_zero_point: tvm.relay.Expr
        The zero point of the output tensor.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.requantize(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis,
        rounding,
        out_dtype,
    )


def quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    r"""Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : tvm.relay.Expr
        The output zero_point.
    output_scale : tvm.relay.Expr
        The output scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8, int32]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.quantize(data, output_scale, output_zero_point, axis, out_dtype)


def simulated_quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    r"""Simulated Quantize op
    Mimics the quantize op but has more flexibility in valid inputs and always
    outputs the same type as the input. This can be useful for
    calibrating or training a quantized network.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : tvm.relay.Expr
        The output zero_point.
    output_scale : tvm.relay.Expr
        The output scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    out_dtype : string or tvm.relay.Expr
        A string or tensor indicating which datatype to quantize to.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # Convert string dtype to a constant if needed.
    if isinstance(out_dtype, str):
        type_code = SQNN_DTYPE_TO_CODE[out_dtype]
        out_dtype = relay.const(type_code, dtype="int32")
    # Wrap reshapes around qnn parameter tensors to guarantee shape compatibility.
    output_scale = relay.op.reshape(output_scale, [-1])
    output_zero_point = relay.op.reshape(output_zero_point, [-1])
    return _make.simulated_quantize(data, out_dtype, output_scale, output_zero_point, axis)


def dequantize(data, input_scale, input_zero_point, axis=-1):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8, int32].
    input_zero_point : tvm.relay.Expr
        The input zero_point.
    input_scale : tvm.relay.Expr
        The input scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dequantize(data, input_scale, input_zero_point, axis)


def simulated_dequantize(data, input_scale, input_zero_point, axis=-1, in_dtype="int8"):
    r"""Simulated Dequantize op
    Mimics the dequantize op but has more flexibility in valid inputs and always
    outputs the same type as the input. This can be useful for calibrating or
    training a quantized network.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized.
    input_zero_point : tvm.relay.Expr
        The input zero_point.
    input_scale : tvm.relay.Expr
        The input scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    in_dtype : string or tvm.relay.Expr
        A string or tensor indicating which datatype to dequantize from.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # Convert string dtype to a constant if needed.
    if isinstance(in_dtype, str):
        type_code = SQNN_DTYPE_TO_CODE[in_dtype]
        in_dtype = relay.const(type_code, dtype="int32")
    # Wrap reshapes around qnn parameter tensors to guarantee shape compatibility.
    input_scale = relay.op.reshape(input_scale, [-1])
    input_zero_point = relay.op.reshape(input_zero_point, [-1])
    return _make.simulated_dequantize(data, in_dtype, input_scale, input_zero_point, axis)


def concatenate(data, input_scales, input_zero_points, output_scale, output_zero_point, axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr], TupleWrapper[relay.Expr])
        The list of quantized tensors.

    input_scales : List[relay.Expr]
        The list of scales of input quantized tensors.

    input_zero_points : List[relay.Expr]
        The list of zero points of input quantized tensors.

    output_scale : relay.Expr
        The scale of the output quantized tensor.

    output_zero_point : relay.Expr
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    if isinstance(data, (list, tuple)):
        data = Tuple(data)
    elif isinstance(data, TupleWrapper):
        data = data.tuple_value
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    input_scales = list(input_scales)
    input_zero_points = list(input_zero_points)

    return _make.concatenate(
        data, Tuple(input_scales), Tuple(input_zero_points), output_scale, output_zero_point, axis
    )


def conv2d(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    channels,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="int32",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel.
    If doing Per-channel quantization, qnn expects the kernel_zero_scale
    and optionally the kernel_zero_point will be 1-D vectors instead of scalars.
    The scale of the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: tvm.relay.Expr
           The zero point of the data distribution.

    kernel_zero_point: tvm.relay.Expr
           The zero point of the quantized_kernel distribution.

    input_scale: tvm.relay.Expr
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: tvm.relay.Expr
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_size : tuple of int
        The spatial width and height of the convolution kernel.

    channels : int
        Number of output channels of this convolution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    # TODO enforce 4-way padding in topi/nn/conv2d after #4644 merged
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d(
        data,
        kernel,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv2d_transpose(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="IOHW",
    out_layout="",
    output_padding=(0, 0),
    out_dtype="int32",
):
    """This operator deconvolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    input_zero_point: tvm.relay.Expr
           The zero point of the data distribution.

    kernel_zero_point: tvm.relay.Expr
           The zero point of the quantized_kernel distribution.

    input_scale: tvm.relay.Expr
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: tvm.relay.Expr
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.transpose_conv2d is lowered to the
           sequence of steps as in nn.transpose_conv2d. See also input_scale in Requantize.

    strides : Tuple[int], optional
        The strides of convolution.

    padding : Tuple[int], optional
        The padding of convolution.

    dilation : Tuple[int], optional
        Specifies the dilation rate to be used for dilated convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial dimensions of the convolution kernel.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    output_padding : Tuple[int], optional
        Used to identify the padding within the output shape
        (only used in training, where transpose_conv represents the gradient of a convolution )

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d_transpose(
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
    )


def add(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.add(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def dense(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    units,
    out_dtype="int32",
):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    If doing Per-channel quantization, qnn expects the kernel_zero_scale
    and optionally the kernel_zero_point will be 1-D vectors instead of scalars.

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: tvm.relay.Expr
        The input zero point.
    kernel_zero_point: tvm.relay.Expr
        The kernel zero point.
    input_scale: tvm.relay.Expr
        The scale for the input tensor.
    kernel_scale: tvm.relay.Expr
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dense(
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        units,
        out_dtype,
    )


def mul(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.mul(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def subtract(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.subtract(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype="int32"):
    r"""
    Computes batch matrix multiplication of `x` and `y` when `x` and `y` are data
    in batch.

    .. math::

        \mbox{batch_matmul}(x, y)[i, :, :] = \mbox{matmul}(x[i, :, :], y[i, :, :]^T)

    Parameters
    ----------
    x : tvm.relay.Expr
        The first quantized input.
        A quantized tensor is represented in following manner
        `A = scale_a x (QA - zp_A)`
        where QA is quantized tensor, scale_a and zp_A are quantization
        params.
    y : tvm.relay.Expr
        The second quantized input.
    x_zero_point: tvm.relay.Expr
        The first input zero point.
    y_zero_point: tvm.relay.Expr
        The second input zero point.
    x_scale: tvm.relay.Expr
        The scale for the first input tensor.
    y_scale: tvm.relay.Expr
        The scale for the second input tensor.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype)


# register fuse pattern for qnn ops
reg.register_pattern("qnn.quantize", OpPattern.OPAQUE)
reg.register_pattern("qnn.dequantize", OpPattern.OPAQUE)
