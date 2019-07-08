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

def quantize(input_data, output_zero_point, output_scale, out_dtype='int8'):
    r""" Quantize op
     This operator takes floating point 32 or quantized int8 and unit8 as input and produces
    quantized int8 or unit8 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.
     ..math::
            \mbox{out}[x] =
                \mbox{clamp(round(input_tensor/output_scale) + output_zero_point); out_dtype::min, out_dtype::max}
     Parameters
    ----------
    input_data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type [float32, int8, uint8].
    output_zero_point :
        The output zero_point.
    output_scale:
        The output scale.
    input_dtype:
        The data type of the input tensor. Can be [int8, uint8, float32]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.quantize(input_data, output_zero_point, output_scale, out_dtype)


def dequantize(input_data, input_zero_point, input_scale):
    r""" Dequantize op
     This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.
     Parameters
    ----------
    input_data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type [float32, int8, uint8].
    input_zero_point :
        The output zero_point.
    input_scale:
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dequantize(input_data, input_zero_point, input_scale)