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
# pylint: disable=unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

from tvm.runtime import convert
from tvm.te.hybrid import script
from topi.util import get_const_int, get_const_tuple
from .op import ShapeDependant, register_reduce_schedule, register_shape_func

register_reduce_schedule("argmax")
register_reduce_schedule("argmin")
register_reduce_schedule("sum")
register_reduce_schedule("all")
register_reduce_schedule("any")
register_reduce_schedule("max")
register_reduce_schedule("min")
register_reduce_schedule("prod")
register_reduce_schedule("mean")
register_reduce_schedule("variance")

def _create_axis_record(attrs, inputs):
    axes = attrs.axis if attrs.axis is None else list(get_const_tuple(attrs.axis))
    exclude = get_const_int(attrs.exclude) > 0
    keepdims = get_const_int(attrs.keepdims) > 0
    data_shape = inputs[0]
    shape_size = data_shape.shape[0].value
    axis_record = [-1] * shape_size
    if axes is None:
        axes = list(range(shape_size))

    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] = shape_size + axis

    if exclude:
        ex_axes = []
        for i in range(shape_size):
            if i not in axes:
                ex_axes.append(i)
        axes = ex_axes

    for i in range(shape_size):
        if i not in axes:
            axis_record[i] = i

    if not keepdims:
        tmp = []
        for i in axis_record:
            if i >= 0:
                tmp.append(i)
        axis_record = tmp

    return axis_record


@script
def _reduce_shape_func(data_shape, axis_record):
    out = output_tensor((len(axis_record),), "int64")
    for i in const_range(len(axis_record)):
        if axis_record[i] >= 0:
            out[i] = data_shape[axis_record[i]]
        else:
            out[i] = int64(1)

    return out

def reduce_shape_func(attrs, inputs, data_inputs, _):
    """
    Shape function for reduce op.
    """
    axis_record = _create_axis_record(attrs, inputs)
    return [_reduce_shape_func(inputs[0], convert(axis_record))]

register_shape_func("argmax", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("argmin", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("all", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("sum", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("max", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("min", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("prod", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("mean", ShapeDependant.SHAPE, reduce_shape_func)
register_shape_func("variance", ShapeDependant.SHAPE, reduce_shape_func)
