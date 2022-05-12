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
"Definition of classic algorithms"
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

from tvm.te.hybrid import script
from tvm.runtime import convert

from . import strategy
from . import op as _reg
from .op import OpPattern, register_pattern
from .op import register_strategy, register_shape_func
from ._tensor import elemwise_shape_func

# sort
register_strategy("sort", strategy.sort_strategy)
register_pattern("sort", OpPattern.OPAQUE)
register_shape_func("sort", False, elemwise_shape_func)

# argsort
register_strategy("argsort", strategy.argsort_strategy)
register_pattern("argsort", OpPattern.OPAQUE)
register_shape_func("argsort", False, elemwise_shape_func)

# topk
register_strategy("topk", strategy.topk_strategy)
register_pattern("topk", OpPattern.OPAQUE)

# searchsorted
register_strategy("searchsorted", strategy.searchsorted_strategy)
register_pattern("searchsorted", OpPattern.OPAQUE)


@script
def _topk_shape_func_input_shape(data_shape, k, axis):
    ndim = data_shape.shape[0]
    val_out = output_tensor((ndim,), "int64")
    indices_out = output_tensor((ndim,), "int64")

    for i in const_range(ndim):
        if i != axis:
            val_out[i] = int64(data_shape[i])
            indices_out[i] = int64(data_shape[i])
        else:
            if k < 1:
                val_out[i] = int64(data_shape[i])
                indices_out[i] = int64(data_shape[i])
            else:
                val_out[i] = int64(k)
                indices_out[i] = int64(k)
    return val_out, indices_out


@_reg.register_shape_func("topk", False)
def topk_shape_func(attrs, inputs, _):
    """
    Shape func for topk.
    """
    axis = attrs.axis
    if axis < 0:
        axis += inputs[0].shape[0]
    val_out, indices_out = _topk_shape_func_input_shape(inputs[0], attrs.k, convert(axis))
    ret_type = attrs.ret_type
    if ret_type == "both":
        ret = [val_out, indices_out]
    elif ret_type == "values":
        ret = [val_out]
    else:
        ret = [indices_out]

    return ret


@script
def _searchsorted_shape(sorted_sequence_shape, values_shape):
    out_shape = output_tensor((values_shape.shape[0],), "int64")
    if sorted_sequence_shape.shape[0] > 1:
        assert (
            sorted_sequence_shape.shape[0] == values_shape.shape[0]
        ), "Ranks of `sorted_sequence` and values must be the same if `sorted_sequence` is not 1-D."
    for i in range(values_shape.shape[0]):
        if sorted_sequence_shape.shape[0] > 1 and i < values_shape.shape[0] - 1:
            assert (
                sorted_sequence_shape[i] == values_shape[i]
            ), "`sorted_sequence and `values` do not have the same shape along outer axes."

        out_shape[i] = values_shape[i]
    return out_shape


@_reg.register_shape_func("searchsorted", False)
def searchsorted_shape_func(attrs, inputs, _):
    """
    Shape func for searchsorted operator.
    """
    return [_searchsorted_shape(inputs[0], inputs[1])]
