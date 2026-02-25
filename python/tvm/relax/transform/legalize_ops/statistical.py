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
"""Default legalization function for statistical operators."""

from typing import List, Tuple, Union

from tvm import te, tir, topi

from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import LegalizeFunc, TEFunc, register_legalize


def _statistical(te_func: TEFunc) -> LegalizeFunc:
    def statistical_call_te(bb: BlockBuilder, call: Call) -> Expr:
        return bb.call_te(te_func, call.args[0], call.attrs.axis, call.attrs.keepdims)

    return statistical_call_te


def _compute_shape_prod(x: te.Tensor, axis: List[tir.IntImm]) -> tir.PrimExpr:
    shape_prod = tir.const(1, "int32")
    axes = [_axis.value for _axis in axis] if axis is not None else range(0, len(x.shape))
    for dim in axes:
        shape_prod = shape_prod * x.shape[dim]
    return shape_prod


def _te_mean(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
    shape_prod = _compute_shape_prod(x, axis)
    res_sum = topi.sum(x, axis, keepdims)
    return topi.divide(res_sum, shape_prod)


def _te_variance(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
    dev = x - _te_mean(x, axis, True)
    return _te_mean(dev * dev, axis, keepdims)
    # This version has better memory locality and performance
    # But may trigger some precision problems, so we will use the previous version now
    # mean = _te_mean(x, axis, keepdims)
    # return _te_mean(x * x, axis, keepdims) - mean * mean


def _te_median(
    x: te.Tensor, axis: List[tir.IntImm], keepdims: bool
) -> Union[te.Tensor, Tuple[te.Tensor, te.Tensor]]:
    # currently only supports one axis or no axis ~ same pytorch
    # todo: support multiple axis ~ same numpy
    shape_prod = _compute_shape_prod(x, axis)
    mid_index = (shape_prod - 1) // 2

    if axis is None or len(axis) == 0:
        x = topi.reshape(x, [shape_prod.value])
        ax = -1
    else:
        ax = axis[0].value
    index_sorted = topi.argsort(x, axis=ax, is_ascend=True, dtype="int64")
    x_sorted = topi.gather(x, axis=ax, indices=index_sorted)

    new_shape = list(x.shape)
    new_shape[ax] = 1
    indices = topi.full(new_shape, fill_value=mid_index, dtype="int64")

    median_val = topi.gather(x_sorted, axis=ax, indices=indices)
    median_idx = topi.gather(index_sorted, axis=ax, indices=indices)

    if axis is None or len(axis) == 0:
        return median_val if keepdims else topi.squeeze(median_val, axis=axis)

    val = median_val
    idx = median_idx
    if not keepdims:
        val = topi.squeeze(val, axis=axis)
        idx = topi.squeeze(idx, axis=axis)
    return val, idx


@register_legalize("relax.mean")
def _mean(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        _te_mean, call.args[0], call.attrs.axis, call.attrs.keepdims, primfunc_name_hint="mean"
    )


@register_legalize("relax.std")
def _std(bb: BlockBuilder, call: Call) -> Expr:
    def te_std(x: te.Tensor, axis: List[tir.IntImm], keepdims: bool) -> te.Tensor:
        return topi.sqrt(_te_variance(x, axis, keepdims))

    return bb.call_te(
        te_std, call.args[0], call.attrs.axis, call.attrs.keepdims, primfunc_name_hint="std"
    )


@register_legalize("relax.variance")
def _variance(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        _te_variance,
        call.args[0],
        call.attrs.axis,
        call.attrs.keepdims,
        primfunc_name_hint="variance",
    )


@register_legalize("relax.median")
def _median(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        _te_median,
        call.args[0],
        call.attrs.axis,
        call.attrs.keepdims,
        primfunc_name_hint="median",
    )


register_legalize("relax.max", _statistical(topi.max))
register_legalize("relax.min", _statistical(topi.min))
register_legalize("relax.prod", _statistical(topi.prod))
register_legalize("relax.sum", _statistical(topi.sum))
