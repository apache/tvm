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
# pylint: disable=invalid-name, too-many-lines
"""Neural network operations."""
from __future__ import absolute_import as _abs
from tvm import relay

def max_pool2d(quantized_data,
               input_zero_point,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False):
    r"""Quantized 2D maximum pooling operator.
    This operator takes quantized data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride
    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:
    with data of shape (b, c, h, w) and pool_size (kh, kw)
    .. math::
        \mbox{out}(b, c, y, x)  = \max_{m=0, \ldots, kh-1} \max_{n=0, \ldots, kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)
    Padding is applied to quantized_data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.
    Parameters
    ----------
    quantized_data : tvm.relay.Expr
        The input quantized_data to the operator.
    input_zero_point: int
       The zero point of the data distribution.
    pool_size : tuple of int, optional
        The size of pooling window.
    strides : tuple of int, optional
        The strides of pooling.
    padding : tuple of int, optional
        The padding for pooling.
    layout : str, optional
        Layout of the input.
    ceil_mode : bool, optional
        To enable or disable ceil while pooling.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    casted_data = relay.cast(quantized_data, dtype="int32")
    shifted_data = relay.subtract(casted_data, relay.const(input_zero_point, "int32"))
    return relay.nn.max_pool2d(shifted_data,
                               pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               layout=layout,
                               ceil_mode=ceil_mode)
