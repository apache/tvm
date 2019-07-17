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


def requantize(data,
               input_scale,
               input_zero_point,
               output_scale,
               output_zero_point,
               out_dtype="int32",
               rounding="FE_AWAY_FROM_ZERO"):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor to another quantized
    tensor. For the output tensor, we are provided with output scale and zero
    point. The computation looks like this

    Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: float
           The float scalar to scale the data int8 values back to FP32.

    input_zero_point: int
           The zero point of the data distribution.

    output_scale: float
           The float scalar to scale the quantized_output int8 values back to FP32.

    output_zero_point: int
           The zero point of the quantized_output distribution.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    assert rounding in ("FE_UPWARD", "FE_AWAY_FROM_ZERO"), \
    "Unsupported rounding mode"

    return _make.requantize(data,
                        input_scale,
                        input_zero_point,
                        output_scale,
                        output_zero_point,
                        out_dtype,
                        rounding)

def quantized_dense(data, weight, input_zero_point, kernel_zero_point, units=None, out_dtype="int32"):
    """Dense operator.
    Applies a linear transformation

    .. math::

    `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantied input data to the operator.

    weight : tvm.relay.Expr
        The quantized weight expressions.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dense(data, weight, units, input_zero_point, kernel_zero_point, out_dtype)
