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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Transposed 1D convolution operators (sometimes called Deconvolution)."""
from __future__ import absolute_import as _abs
import tvm
from .dilate import dilate
from .pad import pad
from ..util import simplify
from .util import get_pad_tuple1d


@tvm.target.generic_func
def conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype):
    """Transposed 1D convolution ncw forward operator.

    Parameters
    ----------
    data : tvm.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        3-D with shape [batch, out_channel, out_width]
    """

    # dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    batch, channels_in, data_width = data.shape
    _, channels_out, kernel_width = kernel.shape
    channels_out = simplify(channels_out)
    data = dilate(data, [1, 1, stride], name='data_dilate')
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right
    data = pad(data, [0, 0, pad_left], [0, 0, pad_right], name='data_pad')

    # transpose kernel, switch kernel layout to IOW
    kernel = tvm.compute((channels_out, channels_in, kernel_width), \
                         lambda o, i, w: kernel[i][o][kernel_width-1-w],\
                         name='kernel')

    # convolution
    _, _, data_width = data.shape
    out_w = simplify(data_width - kernel_width + 1)
    dc = tvm.reduce_axis((0, channels_in), name='dc')
    dw = tvm.reduce_axis((0, kernel_width), name='dw')
    output = tvm.compute(
        (batch, channels_out, out_w),
        lambda b, c, w: tvm.sum(
            data[b, dc, w+dw].astype(out_dtype) *
            kernel[c, dc, dw].astype(out_dtype),
            axis=[dc, dw]), tag="conv1d_transpose_ncw")

    return output
