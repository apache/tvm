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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Argwhere operator"""
import tvm
from tvm.te import hybrid


@hybrid.script
def hybrid_argwhere_1d(output_shape, condition):
    """Find the indices of elements of a 1-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        1-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    a1 = condition.shape[0]
    valid_index = 0
    for i1 in range(a1):
        if condition[i1] != 0:
            a[valid_index, 0] = i1
            valid_index += 1
    return a


@hybrid.script
def hybrid_argwhere_2d(output_shape, condition):
    """Find the indices of elements of a 2-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        2-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            if condition[i1, i2] != 0:
                a[valid_index, 0] = i1
                a[valid_index, 1] = i2
                valid_index += 1
    return a


@hybrid.script
def hybrid_argwhere_3d(output_shape, condition):
    """Find the indices of elements of a 3-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        3-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            for i3 in range(a3):
                if condition[i1, i2, i3] != 0:
                    a[valid_index, 0] = i1
                    a[valid_index, 1] = i2
                    a[valid_index, 2] = i3
                    valid_index += 1
    return a


@hybrid.script
def hybrid_argwhere_4d(output_shape, condition):
    """Find the indices of elements of a 4-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        4-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    a4 = condition.shape[3]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            for i3 in range(a3):
                for i4 in range(a4):
                    if condition[i1, i2, i3, i4] != 0:
                        a[valid_index, 0] = i1
                        a[valid_index, 1] = i2
                        a[valid_index, 2] = i3
                        a[valid_index, 3] = i4
                        valid_index += 1
    return a


@hybrid.script
def hybrid_argwhere_5d(output_shape, condition):
    """Find the indices of elements of a 5-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        5-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    a4 = condition.shape[3]
    a5 = condition.shape[4]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            for i3 in range(a3):
                for i4 in range(a4):
                    for i5 in range(a5):
                        if condition[i1, i2, i3, i4, i5] != 0:
                            a[valid_index, 0] = i1
                            a[valid_index, 1] = i2
                            a[valid_index, 2] = i3
                            a[valid_index, 3] = i4
                            a[valid_index, 4] = i5
                            valid_index += 1
    return a


@tvm.target.generic_func
def argwhere(output_shape, condition):
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    if len(condition.shape) == 1:
        return hybrid_argwhere_1d(output_shape.shape, condition)
    if len(condition.shape) == 2:
        return hybrid_argwhere_2d(output_shape.shape, condition)
    if len(condition.shape) == 3:
        return hybrid_argwhere_3d(output_shape.shape, condition)
    if len(condition.shape) == 4:
        return hybrid_argwhere_4d(output_shape.shape, condition)
    if len(condition.shape) == 5:
        return hybrid_argwhere_5d(output_shape.shape, condition)
    raise ValueError("Does not support rank higher than 5 in argwhere")
