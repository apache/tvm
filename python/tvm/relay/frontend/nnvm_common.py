# pylint: disable=invalid-name, import-self, len-as-condition
"""Utility functions common to NNVM and MxNet conversion."""
from __future__ import absolute_import as _abs

from .. import expr as _expr
from .. import op as _op

def _get_relay_op(op_name):
    op = _op
    for path in op_name.split("."):
        op = getattr(op, path)
    if not op:
        raise RuntimeError("Unable to map op_name {} to relay".format(op_name))
    return op


def _warn_not_used(attr, op='nnvm'):
    import warnings
    err = "{} is ignored in {}.".format(attr, op)
    warnings.warn(err)


def _rename(new_op):
    if isinstance(new_op, str):
        new_op = _get_relay_op(new_op)
    # attrs are ignored.
    def impl(inputs, _, _dtype='float32'):
        return new_op(*inputs)
    return impl


def _reshape(inputs, attrs):
    if attrs.get_bool("reverse", False):
        raise RuntimeError("reshape do not support option reverse")
    shape = attrs.get_int_tuple("shape")
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
    def _impl(inputs, attrs):
        assert len(inputs) == 1
        axis = attrs.get_int("axis", -1)
        return new_op(inputs[0], axis=axis)
    return _impl


def _reduce(new_op):
    """Reduction ops like sum/min/max"""
    def _impl(inputs, attrs):
        assert len(inputs) == 1
        axis = attrs.get_int_tuple("axis", [])
        keepdims = attrs.get_bool("keepdims", False)
        # use None for reduce over all axis.
        axis = None if len(axis) == 0 else axis
        return new_op(inputs[0], axis=axis, keepdims=keepdims)
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


def _elemwise_sum(inputs, _):
    assert len(inputs) > 0
    res = inputs[0]
    for x in inputs[1:]:
        res = _op.add(res, x)
    return res


def _binop_scalar(new_op):
    def _impl(inputs, attrs):
        assert len(inputs) == 1
        scalar = attrs.get_float("scalar")
        # Note: binary scalar only works for float op for now
        scalar = _expr.const(scalar, dtype="float32")
        return new_op(inputs[0], scalar)
    return _impl


def _rbinop_scalar(new_op):
    def _impl(inputs, attrs):
        assert len(inputs) == 1
        scalar = attrs.get_float("scalar")
        # Note: binary scalar only works for float op for now
        scalar = _expr.const(scalar, dtype="float32")
        return new_op(scalar, inputs[0])
    return _impl
