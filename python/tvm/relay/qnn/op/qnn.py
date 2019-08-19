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
#pylint: disable=invalid-name
"""QNN dialect operators."""

from __future__ import absolute_import as _abs
from tvm import relay
from . import _make

def requantize(data,
               input_scale,
               input_zero_point,
               output_scale,
               output_zero_point,
               rounding="TONEAREST",
               out_dtype="int8"):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: float
        The quantization scale for the input tensor.

    input_zero_point: int
        The zero point of the input tensor.

    output_scale: float
        The quantization scale for the output tensor.

    output_zero_point: int
        The zero point of the output tensor.

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

    return _make.requantize(data,
                            input_scale,
                            input_zero_point,
                            output_scale,
                            output_zero_point,
                            rounding,
                            out_dtype)


def quantize(data,
             output_scale,
             output_zero_point,
             out_dtype='int8'):
    r""" Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : int
        The output zero_point.
    output_scale : float
        The output scale.
    input_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.quantize(data,
                          output_scale,
                          output_zero_point,
                          out_dtype)


def dequantize(data,
               input_scale,
               input_zero_point):
    r""" Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : int
        The output zero_point.
    input_scale : float
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dequantize(data,
                            input_scale,
                            input_zero_point)
def concatenate(data,
                input_scales,
                input_zero_points,
                output_scale,
                output_zero_point,
                axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        The list of quantized tensors.

    input_scales : List[float32]
        The list of scales of input quantized tensors.

    input_zero_points : List[int32]
        The list of zero points of input quantized tensors.

    output_scale : float32
        The scale of the output quantized tensor.

    output_zero_point : int32
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    data = list(data)
    requantized_exprs = list(data)

    # Find the dtype of the input expr. This is required for the requantize op. Since, this is
    # concatenate op, the dtype of the input is same as dtype of the output.
    mod = relay.Module.from_expr(data[0])
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    data0 = entry if isinstance(data[0], relay.Function) else entry.body
    in_dtype = data0.checked_type.dtype

    # First check if all the input qnn params match. If yes, we can call concatenate first, followed
    # by a requantize.
    if all(scale == input_scales[0] for scale in input_scales)\
            and all(zero_point == input_zero_points[0] for zero_point in input_zero_points):
        out = relay.concatenate(tuple(data), axis)
        input_scale = input_scales[0]
        input_zero_point = input_zero_points[0]
        if input_scale != output_scale or input_zero_point != output_zero_point:
            out = requantize(data=out,
                             input_scale=input_scales[0],
                             input_zero_point=input_zero_points[0],
                             output_scale=output_scale,
                             output_zero_point=output_zero_point,
                             out_dtype=in_dtype)
        return out

    # If the output qnn params do not match the input qnn params, we can call requantize on the
    # input expr first, followed by a concatenate on the requantized input exprs.
    for idx, quantized_expr in enumerate(data):
        input_scale = input_scales[idx]
        input_zero_point = input_zero_points[idx]
        if input_scale != output_scale or input_zero_point != output_zero_point:
            requantized_exprs[idx] = requantize(data=quantized_expr,
                                                input_scale=input_scale,
                                                input_zero_point=input_zero_point,
                                                output_scale=output_scale,
                                                output_zero_point=output_zero_point,
                                                out_dtype=in_dtype)
    return relay.concatenate(tuple(requantized_exprs), axis)
