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
# pylint: disable=invalid-name, import-self, len-as-condition
"""Utility functions common to NNVM and MxNet conversion."""
from __future__ import absolute_import as _abs

from .. import expr as _expr
from .. import op as _op
from .common import get_relay_op
from .common import infer_type as _infer_type

def _warn_not_used(attr, op='nnvm'):
    import warnings
    err = "{} is ignored in {}.".format(attr, op)
    warnings.warn(err)


def _rename(new_op):
    if isinstance(new_op, str):
        new_op = get_relay_op(new_op)
    # attrs are ignored.
    def impl(inputs, _, _dtype='float32'):
        return new_op(*inputs)
    return impl


def _reshape(inputs, attrs):
    shape = attrs.get_int_tuple("shape")
    reverse = attrs.get_bool("reverse", False)
    if reverse:
        return _op.reverse_reshape(inputs[0], newshape=shape)
    return _op.reshape(inputs[0], newshape=shape)


def _init_op(new_op):
    """Init ops like zeros/ones"""
    def _impl(inputs, attrs):
        assert len(inputs) == 0
        shape = attrs.get_int_tuple("shape")
        dtype = attrs.get_str("dtype", "float32")
        return new_op(shape=shape, dtype=dtype)
    return _impl


def _softmax_op(new_op):
    """softmax/log_softmax"""
    def _impl(inputs, attrs, _dtype='float32'):
        assert len(inputs) == 1
        axis = attrs.get_int("axis", -1)
        return new_op(inputs[0], axis=axis)
    return _impl


def _reduce(new_op):
    """Reduction ops like sum/min/max"""
    def _impl(inputs, attrs, _dtype='float32'):
        assert len(inputs) == 1
        axis = attrs.get_int_tuple("axis", [])
        keepdims = attrs.get_bool("keepdims", False)
        exclude = attrs.get_bool("exclude", False)
        # use None for reduce over all axis.
        axis = None if len(axis) == 0 else axis
        return new_op(inputs[0], axis=axis, keepdims=keepdims, exclude=exclude)
    return _impl


def _arg_reduce(new_op):
    """Arg Reduction ops like argmin/argmax"""
    def _impl(inputs, attrs):
        assert len(inputs) == 1
        axis = attrs.get_int("axis", None)
        keepdims = attrs.get_bool("keepdims", False)
        res = new_op(inputs[0], axis=[axis], keepdims=keepdims)
        # cast to dtype.
        res = res.astype("float32")
        return res
    return _impl


def _cast(inputs, attrs):
    """Type cast"""
    dtype = attrs.get_str("dtype")
    return inputs[0].astype(dtype=dtype)


def _clip(inputs, attrs):
    a_min = attrs.get_float("a_min")
    a_max = attrs.get_float("a_max")
    return _op.clip(inputs[0], a_min=a_min, a_max=a_max)


def _transpose(inputs, attrs):
    axes = attrs.get_int_tuple("axes", None)
    # translate default case
    axes = None if len(axes) == 0 else axes
    return _op.transpose(inputs[0], axes=axes)


def _upsampling(inputs, attrs):
    scale = attrs.get_int("scale")
    return _op.nn.upsampling(inputs[0], scale=scale)


def _elemwise_sum(inputs, _, _dtype='float32'):
    assert len(inputs) > 0
    res = inputs[0]
    for x in inputs[1:]:
        res = _op.add(res, x)
    return res


def _binop_scalar(new_op):
    def _impl(inputs, attrs, odtype=None):
        assert len(inputs) == 1
        scalar = attrs.get_float("scalar")
        if odtype is None:
            odtype = _infer_type(inputs[0]).checked_type.dtype
        scalar = _expr.const(scalar, dtype=odtype)
        return new_op(inputs[0], scalar)
    return _impl


def _rbinop_scalar(new_op):
    def _impl(inputs, attrs, odtype=None):
        assert len(inputs) == 1
        scalar = attrs.get_float("scalar")
        if odtype is None:
            odtype = _infer_type(inputs[0]).checked_type.dtype
        scalar = _expr.const(scalar, dtype=odtype)
        return new_op(scalar, inputs[0])
    return _impl


def _compare(new_op):
    """Compare ops like greater/less"""
    def _impl(inputs, _, odtype='float32'):
        assert len(inputs) == 2
        return new_op(inputs[0], inputs[1]).astype(odtype)
    return _impl
