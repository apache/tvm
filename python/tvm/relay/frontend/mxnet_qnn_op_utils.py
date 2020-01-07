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

zero_centered_uint8_quantized_range = np.float32(255)
zero_centered_int8_quantized_range = np.float32(127)


def _get_mkldnn_scale(data_min,
                      data_max,
                      quantized_range):
    r"""Computes the scale as per MKLDNN specification mentioned here -
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
    real_range = np.max([np.abs(np.float32(data_min)),
                         np.abs(np.float32(data_max))])
    scale = np.divide(quantized_range, real_range)
    scale_inverse = np.divide(1.0, scale)
    return scale_inverse


def _quantize_scale_with_zero_centered(data,
                                       scale,
                                       zero_point,
                                       out_dtype):
    quantized_output = quantize(data,
                                relay.const(scale, 'float32'),
                                relay.const(zero_point ,'int32'),
                                out_dtype=out_dtype)
    return quantized_output, scale, zero_point


def _quantize_with_zero_centered(data,
                                 data_min,
                                 data_max,
                                 quantized_range,
                                 out_dtype):
    r"""Quantizes the given data tensor by calculating the scale
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

    scale = _get_mkldnn_scale(data_min,
                              data_max,
                              quantized_range)
    zero_point = 0
    return _quantize_scale_with_zero_centered(data,
                                              scale,
                                              zero_point,
                                              out_dtype)



def _quantize_mxnet_min_max_uint8(data,
                                  imin_range,
                                  imax_range):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `uint8`.
    The method of quantizing is described here - https://tinyurl.com/y4d7hrzf.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
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

    iinfo = np.iinfo(np.uint8)
    min_limit = np.float64(iinfo.min)
    max_limit = np.float64(iinfo.max)
    imin_range = np.float64(imin_range)
    imax_range = np.float64(imax_range)
    scale = np.divide((max_limit - min_limit),
                      (imax_range - imin_range))
    scale_inverse = np.divide(1.0, scale)
    zero_point = np.int(-1 * imin_range * scale)
    quantized_output = quantize(data,
                                scale_inverse,
                                zero_point,
                                out_dtype='uint8')
    return quantized_output, scale, zero_point


def _quantize_mxnet_min_max_int8(data,
                                 data_min,
                                 data_max):
    r"""Quantizes the given `data` in float32 and the given
    min and max ranges and the output data type is `int8`.
    The method of quantizing is described here - https://tinyurl.com/y4d7hrzf.
    We use our default quantize implementation from src/relay/qnn/op/quantize.cc:72
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
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

    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_int8_quantized_range,
                                        'int8')


def _quantize_mkldnn_min_max_uint8(data,
                                   data_min,
                                   data_max):
    r"""Quantizes the given `data` in float32 and the given
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
    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_uint8_quantized_range,
                                        'uint8')


def _quantize_mkldnn_min_max_int8(data,
                                  data_min,
                                  data_max):
    r"""Quantizes the given `data` in float32 and the given
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
    imin_range : float
        The minimum to use data elements.
    imax_range : float
        The maximum to use for data elements.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _quantize_with_zero_centered(data,
                                        data_min,
                                        data_max,
                                        zero_centered_int8_quantized_range,
                                        'int8')


def get_mkldnn_int8_scale(range_min,
                          range_max):
    r"""Computes the quantization scale using MKLDNN specifications
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

    scale = _get_mkldnn_scale(range_min,
                              range_max,
                              zero_centered_int8_quantized_range)
    return np.float32(scale)


def get_mkldnn_uint8_scale(range_min,
                           range_max):
    r"""Computes the quantization scale using MKLDNN specifications
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

    scale = _get_mkldnn_scale(range_min,
                              range_max,
                              zero_centered_uint8_quantized_range)
    return np.float32(scale)


def quantize_conv_weights_mkldnn_from_var(weights_var,
                                          min_range,
                                          max_range):
    r"""Helper method to quantize the convolution kernel in prequantized model
    in MXNet with MKLDNN. The kernel is always quantized to int8 output datatype.
    The inputs are the raw weights which are floating point numbers. The min and
    max ranges are used from the weight itself. The name supplied is used to create
    a tvm.relay.var with the given name.

    Parameters
    ----------
    weights_var : tvm.relay.var
                The float32 representation of the weights.
    min_range : float32
              A number representing the minimum of the weights.
    max_range : float32
              A number representing the maximum of the weights.

    Returns
    -------
    result : tvm.relay.expr
           The quantized representation of the weights.
    """

    return quantize_mxnet_min_max(weights_var,
                                  min_range,
                                  max_range,
                                  # mkldnn uses only int8 for weights
                                  out_dtype='int8',
                                  use_mkldnn=True)


def get_mkldnn_requantize_scale_outDtype(min_output_range,
                                         max_output_range,
                                         data_scale,
                                         kernel_scale,
                                         out_dtype):
    quantized_out_range = zero_centered_int8_quantized_range if out_dtype == 'int8' \
        else zero_centered_uint8_quantized_range
    out_range = np.max([np.abs(np.float32(min_output_range)),
                        np.abs(np.float32(max_output_range))])
    output_scale = quantized_out_range / out_range
    requantize_scale = np.float32(1/output_scale)
    return requantize_scale


# TODO: add support for uint8 type
def get_conv_mkldnn_requantized_scale_outDtype(min_output_range,
                                               max_output_range,
                                               data_scale,
                                               kernel_scale):
    out_dtype = 'uint8' if min_output_range >= 0.0 else 'int8'
    requantize_scale = get_mkldnn_requantize_scale_outDtype(min_output_range,
                                                            max_output_range,
                                                            data_scale,
                                                            kernel_scale,
                                                            out_dtype)
    return requantize_scale, out_dtype


def quantize_conv_bias_mkldnn_from_var(bias_var,
                                       bias_scale):
    zero_point = 0
    quantized_bias, _, _ = _quantize_scale_with_zero_centered(bias_var,
                                                              bias_scale,
                                                              zero_point,
                                                              'int32')
    return quantized_bias


def quantize_conv_bias_mkldnn(bias,
                              bias_scale,
                              bias_name):
    shape = bias.shape
    bias_data = relay.var(bias_name, shape=shape, dtype='float32')
    return quantize_conv_bias_mkldnn_from_var(bias_data, bias_scale)


def quantize_conv_weights_mkldnn(weights,
                                 weights_name):
    r"""Helper method to quantize the convolution kernel in prequantized model
    in MXNet with MKLDNN. The kernel is always quantized to int8 output datatype.
    The inputs are the raw weights which are floating point numbers. The min and
    max ranges are used from the weight itself. The name supplied is used to create
    a tvm.relay.var with the given name.

    Parameters
    ----------
    weights : float32 tensor
                The float32 representation of the weights.
    weights_name : string
              Will create a tvm.relay.var by this name.

    Returns
    -------
    result : tvm.relay.expr
           The quantized representation of the weights.
    """
    shape = weights.shape
    input_data = relay.var(weights_name, shape=shape, dtype='float32')
    min_range = np.amin(weights)
    max_range = np.amax(weights)
    return quantize_conv_weights_mkldnn_from_var(input_data,
                                                 min_range,
                                                 max_range)


def quantize_mxnet_min_max(data,
                           min_range,
                           max_range,
                           out_dtype='int8',
                           use_mkldnn=False):
    r"""Quantizes the given `data` in float32 and the given
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
    use_mkldnn: bool, optional
        If True then uses MKLDNN quantization implementation otherwise
        will use default implementation.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    if out_dtype == 'uint8':
        if use_mkldnn:
            return _quantize_mkldnn_min_max_uint8(data,
                                                  min_range,
                                                  max_range)
        else:
            return _quantize_mxnet_min_max_uint8(data,
                                                 min_range,
                                                 max_range)
    elif out_dtype == 'int8':
        if use_mkldnn:
            return _quantize_mkldnn_min_max_int8(data,
                                                 min_range,
                                                 max_range)
        else:
            return _quantize_mxnet_min_max_int8(data,
                                                min_range,
                                                max_range)
    else:
        raise ValueError(
            "Expected out_dtype to be int8 or uint8 but was  %s" % out_dtype)


def _dequantize_zero_centered(data,
                              data_min,
                              data_max,
                              quantized_range):
    r"""Dequantizes the given data tensor by calculating the scale
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

    real_range = np.max([np.abs(np.float32(data_min)),
                         np.abs(np.float32(data_max))])
    scale = relay.const(np.divide(real_range, quantized_range), 'float32')
    zero_point = relay.const(0, 'int32')
    return dequantize(data, scale, zero_point)


def _dequantize_mkldnn_min_max_int8(data,
                                    imin_range,
                                    imax_range):
    r"""Dequantizes the given `data` in {int8 or uint8} and the given
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

    return _dequantize_zero_centered(data,
                                     data_min=imin_range,
                                     data_max=imax_range,
                                     quantized_range=zero_centered_int8_quantized_range)


def _dequantize_mkldnn_min_max_uint8(data,
                                     imin_range,
                                     imax_range):
    r"""Dequantizes the given `data` in {int8 or uint8} and the given
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

    return _dequantize_zero_centered(data,
                                     data_min=imin_range,
                                     data_max=imax_range,
                                     quantized_range=zero_centered_uint8_quantized_range)


def _dequantize_mxnet_min_max_int8(data,
                                   imin_range,
                                   imax_range):
    r"""Deuantizes the given `data` in {int8 or uint8} and the given
    min and max ranges and the output data type is `float32`.
    The method of dequantization is described here - https://tinyurl.com/y4d7hrzf.
    We use our default dequantize implementation from src/relay/qnn/op/dequantize.cc:67
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
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

    return _dequantize_zero_centered(data,
                                     data_min=imin_range,
                                     data_max=imax_range,
                                     quantized_range=zero_centered_int8_quantized_range)


def _dequantize_mxnet_min_max_uint8(data,
                                    imin_range,
                                    imax_range):
    r"""Dequantizes the given `data` in {int8 or uint8} and the given
    min and max ranges and the output data type is `float32`.
    The method of dequantizing is described here - https://tinyurl.com/y4d7hrzf.
    We use our default quantize implementation from src/relay/qnn/op/dequantize.cc:67
    but compute the `scale` and `zero_point` to fit our equation.
    Unlike in TFLite where we get the scale and zero_point from the model, Mxnet
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

    iinfo = np.iinfo(np.uint8)
    min_limit = np.float64(iinfo.min)
    max_limit = np.float64(iinfo.max)
    imin_range = np.float64(imin_range)
    imax_range = np.float64(imax_range)
    scale_val = np.divide((imax_range - imin_range),
                          (max_limit - min_limit))
    zero_point_val = np.int(-1 * np.divide(imin_range, scale_val))
    scale = relay.const(scale_val, 'float32')
    zero_point = relay.const(zero_point_val, 'int32')
    return dequantize(data, scale, zero_point)


def dequantize_mxnet_min_max(data,
                             min_range,
                             max_range,
                             in_dtype='int8',
                             use_mkldnn=False):
    r"""Dequantizes the given `data` in {int8 or uint8} and the given
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
    use_mkldnn: bool, optional
        If True then uses MKLDNN quantization implementation otherwise
        will use default implementation.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    if in_dtype == 'uint8':
        if use_mkldnn:
            return _dequantize_mkldnn_min_max_uint8(data,
                                                    min_range,
                                                    max_range)
        else:
            return _dequantize_mxnet_min_max_uint8(data,
                                                   min_range,
                                                   max_range)
    elif in_dtype == 'int8':
        if use_mkldnn:
            return _dequantize_mkldnn_min_max_int8(data, min_range, max_range)
        else:
            return _dequantize_mxnet_min_max_int8(data, min_range, max_range)
    else:
        raise ValueError(
            "Expected out_dtype to be int8 or uint8 but was  %s" % in_dtype)


def get_dtype_from_min_max(range_min, range_max):
    assert range_min is not None and range_max is not None
    return 'uint8' if range_min >= 0.0 else 'int8'
