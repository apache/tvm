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
#pylint: disable=invalid-name, too-many-lines
"""Neural network operations."""
from __future__ import absolute_import as _abs
from . import _make


def conv2d(quantized_data,
           quantized_weight,
           input_zero_point,
           kernel_zero_point,
           strides=(1, 1),
           padding=(0, 0),
           dilation=(1, 1),
           groups=1,
           channels=None,
           kernel_size=None,
           data_layout="NCHW",
           kernel_layout="OIHW",
           out_layout="",
           out_dtype="int32"):
    r"""Quantized 2D convolution.

    This operator takes the quantized_weight as the convolution kernel
    and convolves it with quantized_data to produce an output quantized tensor.
    The scale of the output quantized tensor is the prodcut of the weight_scale
    and input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    see Requantize operator to understand the dtype scaling back to (u)int8.

    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a quantized_data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a quantized_weight Tensor with shape `(channels, in_channels, kernel_size[0], kernel_size[1])`
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, y, x] = \sum_{dy, dx, k}
           \mbox{quantized_data}[b, k, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] *
           \mbox{quantized_weight}[c, k, dy, dx]

    Padding and dilation are applied to quantized_data and quantized_weight respectively before the computation.
    This operator accepts quantized_data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCHW` for quantized_data and `OIHW` for quantized_weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    quantized_data : tvm.relay.Expr
        The input quantized_data to the operator.

    quantized_weight : tvm.relay.Expr
        The quantized_weight expressions.

    input_zero_point: int
           The zero point of the quantized_data distribution.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the quantized_weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output quantized_data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.conv2d(quantized_data, quantized_weight,
                        input_zero_point, kernel_zero_point,
                        strides, padding, dilation,
                        groups, channels, kernel_size,
                        data_layout, kernel_layout, out_layout, out_dtype)

def requantize(input_data, input_zero_point, input_scale, output_zero_point,
        output_scale, out_dtype="int32", use_int_compute=False,
        rounding_mode="FE_UPWARD"):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor to another quantized
    tensor. For the output tensor, we are provided with output scale and zero
    point. The computation looks like this

    Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)

    The above computation can be done in floating point as the scales are in
    FP32. Alternatively, we can approximate floating point with fixed point
    computation. This is controlled by use_int_compute.

    Parameters
    ----------
    quantized_data : tvm.relay.Expr
        The input quantized_data to the operator.

    input_scale: float
           The float scalar to scale the quantized_data int8 values back to FP32.

    output_scale: float
           The float scalar to scale the quantized_output int8 values back to FP32.

    input_zero_point: int
           The zero point of the quantized_data distribution.

    output_zero_point: int
           The zero point of the quantized_output distribution.

    out_dtype : str, optional
        Specifies the output quantized_data type for mixed precision conv2d.

    use_int_compute : bool, optional
        Use fully integer computation for requantizing.

    rounding_mode : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    assert rounding_mode in ("FE_UPWARD", "FE_AWAY_FROM_ZERO"),\
            "Unsupported rounding mode"

    return _make.requantize(input_data, input_zero_point, input_scale,
                            output_zero_point, output_scale, out_dtype,
                            use_int_compute, rounding_mode)
