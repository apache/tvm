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
from tvm.relay.qnn.op.qnn import quantize

zero_centered_uint8quantized_range = np.float32(255)
zero_centered_int8quantized_range = np.float32(127)


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
    quantized_output = quantize(data,
                                scale,
                                zero_point,
                                out_dtype=out_dtype)
    return quantized_output, scale, zero_point


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
                                        zero_centered_int8quantized_range,
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
                                        zero_centered_uint8quantized_range,
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
                                        zero_centered_int8quantized_range,
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
                              zero_centered_int8quantized_range)
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
                              zero_centered_uint8quantized_range)
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
