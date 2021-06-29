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
# pylint: disable=invalid-name
"""TVM operator for softmax and log_softmax compute."""
from __future__ import absolute_import
import tvm
from tvm import te, topi


@tvm.te.tag_scope(tag="softmax_output")
def softmax(x, axis=-1):
    """Perform softmax activation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        can be any dimension

    axis : int
        channel axis

    Returns
    -------
    output : tvm.te.Tensor
        output shape is the same as input
    """
    return softmax_common(x, axis, False)


@tvm.te.tag_scope(tag="fast_softmax_output")
def fast_softmax(x, axis=-1):
    """Perform softmax activation on the data.
    Use approximation to compute exponent for faster speed.

    Parameters
    ----------
    data : tvm.te.Tensor
        can be any dimension

    axis : int
        channel axis

    Returns
    -------
    output : tvm.te.Tensor
        output shape is the same as input
    """
    return softmax_common(x, axis, True)


def softmax_common(x, axis, use_fast_exp):
    """The common part of softmax and fast_softmax"""
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    if axis >= len(shape):
        ValueError("axis parameter should be less than input dim")

    k1 = te.reduce_axis((0, shape[axis]), name="k")
    k2 = te.reduce_axis((0, shape[axis]), name="k")

    def insert_reduce_index(indices, reduce_index):
        return indices[:axis] + (reduce_index,) + indices[axis:]

    def get_non_reduce_indices(indices):
        return tuple([var for (i, var) in enumerate(indices) if i != axis])

    def _compute_max(*indices):
        eval_range = insert_reduce_index(indices, k1)
        return tvm.te.max(x[eval_range], axis=k1)

    def _compute_delta(max_elem, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return x[indices] - max_elem[non_reduce_indices]

    def _compute_exp(max_elem, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return te.exp(x[indices] - max_elem[non_reduce_indices])

    def _compute_expsum(exp, *indices):
        eval_range = insert_reduce_index(indices, k2)
        return te.sum(exp[eval_range], axis=k2)

    def _normalize(exp, expsum, *indices):
        non_reduce_indices = get_non_reduce_indices(indices)
        return exp[indices] / expsum[non_reduce_indices]

    reduced_shape = tuple([dim for (i, dim) in enumerate(shape) if i != axis])
    max_elem = te.compute(reduced_shape, _compute_max, name="T_softmax_maxelem")

    if use_fast_exp:
        delta = te.compute(
            shape, lambda *indices: _compute_delta(max_elem, *indices), name="T_softmax_delta"
        )
        exp = topi.math.fast_exp(delta)
    else:
        exp = te.compute(
            shape, lambda *indices: _compute_exp(max_elem, *indices), name="T_softmax_exp"
        )
    expsum = te.compute(
        reduced_shape, lambda *indices: _compute_expsum(exp, *indices), name="T_softmax_expsum"
    )
    return te.compute(
        shape,
        lambda *indices: _normalize(exp, expsum, *indices),
        name="T_softmax_norm",
        attrs={"axis": axis},
    )


@tvm.te.tag_scope(tag="log_softmax_output")
def log_softmax(x, axis=-1):
    """Perform log softmax activation on the data

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D input data

    Returns
    -------
    output : tvm.te.Tensor
        2-D output with same shape
    """
    assert len(x.shape) == 2, "only support 2-dim log softmax"
    # pylint: disable=R1714
    assert axis == -1 or axis == len(x.shape) - 1, "only support last axis log softmax"
    m, n = x.shape
    k = te.reduce_axis((0, n), name="k")
    max_elem = te.compute((m,), lambda i: tvm.te.max(x[i, k], axis=k))
    k = te.reduce_axis((0, n), name="k")
    expsum = te.compute((m,), lambda i: te.sum(te.exp(x[i, k] - max_elem[i]), axis=k))
    return te.compute(x.shape, lambda i, j: x[i, j] - max_elem[i] - te.log(expsum[i]))
