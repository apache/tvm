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
"""Contrib operations."""
from __future__ import absolute_import as _abs
from . import _make


def adaptive_max_pool2d(data,
                        output_size=None,
                        layout="NCHW"):
    r"""2D adaptive max pooling operator. This operator is experimental.

    This operator takes data as input and does 2D max value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size) for any input (NCHW).

        If a tuple of integers (height, width) are provided for output_size,
        the output size is (N x C x height x width) for any input (NCHW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_max_pool2d(data, output_size, layout)

def adaptive_avg_pool2d(data,
                        output_size=None,
                        layout="NCHW"):
    r"""2D adaptive average pooling operator. This operator is experimental.

    This operator takes data as input and does 2D average value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size) for any input (NCHW).

        If a tuple of integers (height, width) are provided for output_size,
        the output size is (N x C x height x width) for any input (NCHW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_avg_pool2d(data, output_size, layout)
