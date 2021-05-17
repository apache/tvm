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
# pylint: disable=invalid-name,unused-argument
"""TVM operator negative log likelihood loss compute."""
from __future__ import absolute_import
from . import cpp


def nll_loss(input, target, weight, reduction, ignore_index):
    """Negative log likelihood loss on the input data.

    Parameters
    ----------
    input : tvm.te.Tensor
        (k+2)-D with shape (N, C, d_1, d_2, ..., d_k),
        where C is the number of target classes

    target : tvm.te.Tensor
        (k+1)-D with shape (N, d_1, d_2, ..., d_k)
        The target value of the input.

    weight : tvm.te.Tensor
        1-D with shape (C,)
        The weight of each target value.

    reduction : string
        The reduction method to apply to output.
        Can be "mean", "sum" or "none".

    ignore_index : int
        The target value to ignore.

    Returns
    -------
    output : tvm.te.Tensor
        a scalar if the reduction type is "mean" or "sum",
        otherwise the same shape as `target`.
    """
    return cpp.nn.nll_loss(input, target, weight, reduction, ignore_index)
