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
from tvm.relay.qnn.op.qnn import dequantize

zero_centered_uint8_quantized_range = np.float32(255)
zero_centered_int8_quantized_range = np.float32(127)


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
    scale = np.divide(real_range, quantized_range)
    zero_point = 0
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
    scale = np.divide((imax_range - imin_range),
                      (max_limit - min_limit))
    zero_point = np.int(-1 * np.divide(imin_range, scale))
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
