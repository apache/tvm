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
# pylint: disable=invalid-name, import-self, len-as-condition, no-else-return
"""MXNet qnn dialect helper methods for MXNet specific implementations of more
   generic qnn supported ops.
"""

import numpy as np
from tvm import relay
from tvm.relay.qnn.op.qnn import quantize, dequantize

# The below values are taken from -
# https://github.com/apache/incubator-mxnet/blob/master/src/operator/quantization/quantization_utils.h#L38-L39
zero_centered_uint8_quantized_range = np.float32(255.5)
zero_centered_int8_quantized_range = np.float32(127.5)


def _get_mkldnn_scale(data_min, data_max, quantized_range):
    """Computes the scale as per MKLDNN specification mentioned here -
    https://intel.github.io/mkl-dnn/ex_int8_simplenet.html

    Parameters
    ----------
    data_min : float32
        A number representing the lower end of the tensor to be quantized.
    data_max : float32
        A number representing the upper end of the tensor to be quantized.
    quantized_range : float32
        255 for uint8 and 127 for int8. This is the data type range.

    Returns
    -------
    scale : A floating point number which acts as the scale for quantization.
    """
    real_range = np.max([np.abs(np.float32(data_min)), np.abs(np.float32(data_max))])
    scale = np.divide(quantized_range, real_range)
    scale_inverse = np.divide(1.0, scale)
    return scale_inverse


def _quantize_scale_with_zero_centered(data, scale, zero_point, out_dtype):
    quantized_output = quantize(
        data, relay.const(scale, "float32"), relay.const(zero_point, "int32"), out_dtype=out_dtype
    )
    return quantized_output, scale, zero_point


def _quantize_with_zero_centered(data, data_min, data_max, quantized_range, out_dtype):
    """Quantizes the given data tensor by calculating the scale
    using the MKLDNN formula `quantized_range / max(abs(data_min, data_max))`.
    Where quantized_range is 255 for uint8 and 127 for int8. The `data_min`
    and `data_max` are the min and max to use for the `data` tensor elements.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    data_min : float
        The minimum to use data elements.
    data_max : float
        The maximum to use for data elements.
    quantized_range : float
        255 for uint8 and 127 for int8. This is the data type range.
    out_dtype : str
        The output data type. Can be int8 or uint8

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    scale = _get_mkldnn_scale(data_min, data_max, quantized_range)
    zero_point = 0
    return _quantize_scale_with_zero_centered(data, scale, zero_point, out_dtype)


def _quantize_mkldnn_min_max_uint8(data, data_min, data_max):
    """Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `uint8`.
    The method of quantizing is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _quantize_with_zero_centered(
        data, data_min, data_max, zero_centered_uint8_quantized_range, "uint8"
    )


def _quantize_mkldnn_min_max_int8(data, data_min, data_max):
    """Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `int8`.
    The method of quantizing is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    data_min : float
        The minimum to use data elements.
    data_max : float
        The maximum to use for data elements.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _quantize_with_zero_centered(
        data, data_min, data_max, zero_centered_int8_quantized_range, "int8"
    )


def get_mkldnn_int8_scale(range_min, range_max):
    """Computes the quantization scale using MKLDNN specifications
    with the given range. The output datatype of tensor to be quantized should be
    int8.

    Parameters
    ----------
    range_min : float32
        A number representing the lower end of the tensor to be quantized.
    range_max : float32
        A number representing the upper end of the tensor to be quantized.

    Returns
    -------
    scale : A float32 number which acts as the scale for quantization.
    """

    scale = _get_mkldnn_scale(range_min, range_max, zero_centered_int8_quantized_range)
    return np.float32(scale)


def get_mkldnn_uint8_scale(range_min, range_max):
    """Computes the quantization scale using MKLDNN specifications
    with the given range. The output datatype of tensor to be quantized should be
    uint8.

    Parameters
    ----------
    range_min : float32
        A number representing the lower end of the tensor to be quantized.
    range_max : float32
        A number representing the upper end of the tensor to be quantized.

    Returns
    -------
    scale : A float32 number which acts as the scale for quantization.
    """

    scale = _get_mkldnn_scale(range_min, range_max, zero_centered_uint8_quantized_range)
    return np.float32(scale)


def quantize_conv_weights_bias_channel_mkldnn_from_var(
    weights_var, bias, min_vector_range, max_vector_range, data_scale
):
    """Helper method to quantize the convolution kernel in prequantized model
    in MXNet with MKLDNN. The kernel is always quantized to int8 output datatype.
    The inputs are the raw weights which are floating point numbers. The min and
    max ranges are used from the weight itself. The name supplied is used to create
    a tvm.relay.var with the given name.

    Parameters
    ----------
    weights_var : tvm.relay.var
        The float32 representation of the weights.
    bias : np.array
        The float32 np array for bias.
    min_vector_range : array of float32
        A number representing the minimum of the weights per channel.
    max_vector_range : array of float32
        A number representing the maximum of the weights per channel.
    data_scale : float
        The data scale value.

    Returns
    -------
    result : tvm.relay.expr
           The quantized representation of the weights.
    """

    quantized_range = zero_centered_int8_quantized_range
    real_vector_range = np.maximum(np.absolute(min_vector_range), np.absolute(max_vector_range))
    # If real_vector_range is 0, then to avoid division by 0 in scaling,
    # make real_vector INT32_max
    vector_scale = np.where(
        real_vector_range == 0,
        1.0 / float(np.iinfo(np.int32).max),
        np.divide(real_vector_range, quantized_range),
    )

    # Handle bias impact on scales as done by MxNet-MKLDNN.
    if bias is not None:
        common = 2.0 * bias.astype("float32") * (1 / data_scale)
        vector_scale_min = np.where(
            bias > 0, common / float(np.iinfo(np.int32).max), common / float(np.iinfo(np.int32).min)
        )
        vector_scale = np.maximum(vector_scale, vector_scale_min)

    zero_point = 0
    quantized_output = quantize(
        weights_var,
        relay.const(vector_scale),
        relay.const(zero_point, "int32"),
        axis=0,
        out_dtype="int8",
    )
    return quantized_output, vector_scale, zero_point


def get_mkldnn_requantize_scale_outDtype(min_output_range, max_output_range, out_dtype):
    """Get the MKLDNN requantized scale."""
    quantized_out_range = (
        zero_centered_int8_quantized_range
        if out_dtype == "int8"
        else zero_centered_uint8_quantized_range
    )
    out_range = np.max([np.abs(np.float32(min_output_range)), np.abs(np.float32(max_output_range))])
    output_scale = quantized_out_range / out_range
    requantize_scale = np.float32(1 / output_scale)
    return requantize_scale


def get_conv_mkldnn_requantized_scale_outDtype(min_output_range, max_output_range):
    out_dtype = "uint8" if min_output_range >= 0.0 else "int8"
    requantize_scale = get_mkldnn_requantize_scale_outDtype(
        min_output_range, max_output_range, out_dtype
    )
    return requantize_scale, out_dtype


def quantize_conv_bias_mkldnn_from_var(bias_var, bias_scale):
    """Quantized conv2d bias"""
    zero_point = 0
    quantized_bias = quantize(
        data=bias_var,
        output_scale=relay.const(bias_scale),
        output_zero_point=relay.const(zero_point, "int32"),
        axis=0,
        out_dtype="int32",
    )

    return quantized_bias


def quantize_mxnet_min_max(data, min_range, max_range, out_dtype="int8"):
    """Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type.
    Only `int8` and `uint8` is supported as output data types.
    The input data type is expected to be `float32`.
    Mxnet has two different flavors for quantization 1) Default 2)MKLDNN.
    To get the second one Mxnet must be built with MKLDNN during compile time.
    Users can choose either of the implementation for TVM runtime.
    The main difference between the two implementation is that MKLDNN is centered
    around 0 and the default implementation for uint8 is not.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    min_range : float
        The minimum to use data elements.
    max_range : float
        The maximum to use for data elements.
    out_dtype: str, optional
        The output data type, can be 'int8' or 'uint8'

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    if out_dtype == "uint8":
        return _quantize_mkldnn_min_max_uint8(data, min_range, max_range)
    elif out_dtype == "int8":
        return _quantize_mkldnn_min_max_int8(data, min_range, max_range)
    else:
        raise ValueError("Expected out_dtype to be int8 or uint8 but was  %s" % out_dtype)


def _dequantize_zero_centered(data, data_min, data_max, quantized_range):
    """Dequantizes the given data tensor by calculating the scale
    using the MKLDNN formula `max(abs(data_min, data_max))/quantized_range`.
    Where quantized_range is 255 for uint8 and 127 for int8. The `data_min`
    and `data_max` are the min and max to use for the `data` tensor elements.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type {int8 or uint8}.
    data_min : float
        The minimum to use data elements.
    data_max : float
        The maximum to use for data elements.
    quantized_range : float
        255 for uint8 and 127 for int8. This is the data type range.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    real_range = np.max([np.abs(np.float32(data_min)), np.abs(np.float32(data_max))])
    scale = relay.const(np.divide(real_range, quantized_range), "float32")
    zero_point = relay.const(0, "int32")
    return dequantize(data, scale, zero_point)


def _dequantize_mkldnn_min_max_int8(data, imin_range, imax_range):
    """Dequantizes the given `data` in {int8 or uint8} and the given
    min and max ranges and the output data type is `float32`.
    The method of dequantizing is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/dequantize.cc:67
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _dequantize_zero_centered(
        data,
        data_min=imin_range,
        data_max=imax_range,
        quantized_range=zero_centered_int8_quantized_range,
    )


def _dequantize_mkldnn_min_max_uint8(data, imin_range, imax_range):
    """Dequantizes the given `data` in {int8 or uint8} and the given
    min and max ranges and the output data type is `float32`.
    The method of dequantize is described here - https://tinyurl.com/y5k6fz5w.
    We use our default quantize implementation from src/relay/qnn/op/dequantize.cc:67
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, MKLDNN
    stores the min and max from which we calculate the scale and zero_point.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _dequantize_zero_centered(
        data,
        data_min=imin_range,
        data_max=imax_range,
        quantized_range=zero_centered_uint8_quantized_range,
    )


def dequantize_mxnet_min_max(data, min_range, max_range, in_dtype="int8"):
    """Dequantizes the given `data` in {int8 or uint8} and the given
    min and max ranges. The output data type is float32.
    Only `float32` is supported as output data types.
    The input data type is expected to be {int8 or uint8}.
    Mxnet has two different flavors for dequantization 1) Default 2)MKLDNN.
    To get the second one Mxnet must be built with MKLDNN during compile time.
    Users can choose either of the implementation for TVM runtime.
    The main difference between the two implementation is that MKLDNN is centered
    around 0 and the default implementation for uint8 is not.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    min_range : float
        The minimum to use data elements for the output.
    max_range : float
        The maximum to use for data elements for the output.
    in_dtype: str, optional
        The input data type, can be 'int8' or 'uint8'

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    if in_dtype == "uint8":
        return _dequantize_mkldnn_min_max_uint8(data, min_range, max_range)
    elif in_dtype == "int8":
        return _dequantize_mkldnn_min_max_int8(data, min_range, max_range)
    else:
        raise ValueError("Expected out_dtype to be int8 or uint8 but was  %s" % in_dtype)
