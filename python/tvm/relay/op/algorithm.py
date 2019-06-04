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
"""Classic algorithm operation"""
from __future__ import absolute_import as _abs
from . import _make
from ..expr import TupleWrapper

def argsort(data, axis=-1, is_ascend=1, dtype="int32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    valid_count : tvm.Tensor
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the output indices.

    Returns
    -------
    out : relay.Expr
        Tensor with same shape as data.
    """
    return _make.argsort(data, axis, is_ascend, dtype)


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int32"):
    """Get the top k elements in an input tensor along the given axis.

    ret_type specifies the return type, can be one of ("both", "values", "indices").

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    k : int, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : relay.Expr or List[relay.Expr]
        The computed result.
    """
    out = _make.topk(data, k, axis, ret_type, is_ascend, dtype)
    if ret_type == "both":
        return TupleWrapper(out, 2)
    return out
