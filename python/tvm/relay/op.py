"""Relay operators"""
from __future__ import absolute_import as _abs

import sys

from .._ffi.node import convert_to_node
from . import _make
from ..make import node as _make_node
from .expr import Expr, Call
from .base import register_relay_node

@register_relay_node
class Op(Expr):
    pass

def _create_op(op_name):
    op = _GetOp(op_name)
    attrs_type_key = op.attrs_type_key
    attrs_type_key = attrs_type_key if attrs_type_key else "DictAttrs"
    # TODO(tqchen): improve the code build to fix the restriction.
    #
    # current restriction:
    # - pass in args as positional arguments
    # - pass in kwargs as keyword argument
    def _op_func(*args, **kwargs):
        args = convert_to_node(args)
        # Need work to make sure constructor matches
        # can support kwargs later
        attrs = _make_node(attrs_type_key,  **kwargs)
        return Call(op, args, None, [])
    _op_func.__name__ = op.name
    return _op_func


def _init_ops():
    """Helper function to initialize the operators
    """
    module = sys.modules[__name__]
    for name in _ListOpNames():
        f = _create_op(name.value)
        setattr(module, f.__name__, f)


_init_ops()
