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
# pylint: disable=unused-variable
"""1D convolution in python"""
import numpy as np
import scipy
import topi
from topi.nn.util import get_pad_tuple1d

def conv1d_ncw_python(a_np, w_np, stride, padding, pad_method, dilation):
    """1D convolution operator in NCW layout

    Parameters
    ----------
    a_np : numpy.ndarray
        3-D with shape [batch, in_channel, in_width]
    
    w_np : numpy.ndarray
        3-D with shape [num_filter, in_channel, filter_width]

    stride : int
        Stride size

    padding : int, tuple, or str
        Single int for padding size or tuple of (left, right) padding
        or a string in ['VALID', 'SAME']

    pad_method : str
        How to pad data, must be in ['SYMMETRIC', 'BEFORE', 'AFTER']

    dilation : int
        Dilation rate of the kernel

    Returns
    -------
    b_np : np.ndarray
        3-D with shape [batch, out_channel, out_width]
    """
    batch, in_c, in_w = a_np.shape
    out_c, _, filter_w = w_np.shape
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    dilated_filter_w = (filter_w - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_filter_w,))
    out_w = (in_w - dilated_filter_w + pad_left + pad_right) // (stride + 1)

    padded_a_np = np.zeros((batch, in_c, in_w + pad_left + pad_right))
    if pad_method == 'SYMMETRIC':
        padded_a_np[:, :, pad_left:(in_w + pad_left)] = a_np
    elif pad_method == 'BEFORE':
        padded_a_np[:, :, (pad_left + pad_right):(in_w + pad_left + pad_right)] = a_np
    elif pad_method == 'AFTER':
        padded_a_np[:, :, :in_w] = a_np
    else:
        raise ValueError("Pad method {} is not supported.".format(pad_method))