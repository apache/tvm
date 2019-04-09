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
# pylint: disable=invalid-name, line-too-long
"""Operators of one-to-one-mapping on the first input"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag

@tvm.tag_scope(tag=tag.BROADCAST)
def scale_shift_nchw(Input, Scale, Shift):
    """Batch normalization operator in inference.

    Parameters
    ----------
    Input : tvm.Tensor
        Input tensor, layout is NCHW

    Scale : tvm.Tensor
        Scale tensor, 1-D of size channel number

    Shift : tvm.Tensor
        Shift tensor, 1-D of size channel number

    Returns
    -------
    Output : tvm.Tensor
        Output tensor, layout is NCHW
    """
    return tvm.compute(Input.shape, lambda b, c, i, j: Input[b, c, i, j] * Scale[c] + Shift[c], name='ScaleShift')


@tvm.tag_scope(tag=tag.BROADCAST)
def scale_shift_nhwc(Input, Scale, Shift):
    """Batch normalization operator in inference.

    Parameters
    ----------
    Input : tvm.Tensor
        Input tensor, layout is NHWC

    Scale : tvm.Tensor
        Scale tensor, 1-D of size channel number

    Shift : tvm.Tensor
        Shift tensor, 1-D of size channel number

    Returns
    -------
    Output : tvm.Tensor
        Output tensor, layout is NHWC
    """
    return tvm.compute(Input.shape, lambda b, i, j, c: Input[b, i, j, c] * Scale[c] + Shift[c], name='ScaleShift')
