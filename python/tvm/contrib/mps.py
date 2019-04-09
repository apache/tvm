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
"""External function interface to MPS libraries."""
from __future__ import absolute_import as _abs
from .. import api as _api
from .. import intrin as _intrin

# pylint: disable=C0103,W0612

def matmul(lhs, rhs, transa=False, transb=False):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to calle external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    m = lhs.shape[0] if transa is False else lhs.shape[1]
    n = rhs.shape[1] if transb is False else rhs.shape[0]
    if transa:
        m = b
    if transb:
        n = c
    return _api.extern(
        (m, n), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.mps.matmul", ins[0], ins[1], outs[0], transa, transb),
        name="C")

def conv2d(data, weight, pad='SAME', stride=1):
    """
    Create an extern op that compute data * weight and return result in output

    Parameters:
    ----------
    data: Tensor
        The input data, format NHWC
    weight: Tensor
        The conv weight, format output_feature * kH * kW * input_feature
    pad: str
        Padding method, 'SAME' or 'VALID'
    stride: int
        convolution stride

    Returns
    -------
    output: Tensor
        The result tensor
    """
    n, hi, wi, ci = data.shape
    co, kh, kw, ciw = weight.shape
    padding = 0 if pad == 'SAME' else 1
    ho = hi // stride
    wo = wi // stride

    return _api.extern(
        (n, ho, wo, co), [data, weight],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.mps.conv2d", ins[0], ins[1], outs[0], padding, stride),
        name="C")
