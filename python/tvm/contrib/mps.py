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
import tvm
from tvm import te


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
    return te.extern(
        (m, n),
        [lhs, rhs],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        name="C",
    )


def conv2d(data, weight, pad="SAME", stride=1):
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
    padding = 0 if pad == "SAME" else 1
    ho = hi // stride
    wo = wi // stride

    return te.extern(
        (n, ho, wo, co),
        [data, weight],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.conv2d", ins[0], ins[1], outs[0], padding, stride
        ),
        name="C",
    )
