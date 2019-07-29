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
