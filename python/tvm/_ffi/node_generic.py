"""Common implementation of Node generic related logic"""
# pylint: disable=unused-import
from __future__ import absolute_import

from numbers import Number, Integral
from .. import _api_internal
from .base import string_types

# Node base class
_CLASS_NODE_BASE = None

def _set_class_node_base(cls):
    global _CLASS_NODE_BASE
    _CLASS_NODE_BASE = cls

class NodeGeneric(object):
    """Base class for all classes that can be converted to node."""
    def asnode(self):
        """Convert value to node"""
        raise NotImplementedError()

def convert_to_node(value):
    """Convert a python value to corresponding node type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    node : Node
        The corresponding node value.
    """
    if isinstance(value, _CLASS_NODE_BASE):
        return value
    elif isinstance(value, bool):
        return const(value, 'uint1x1')
    elif isinstance(value, Number):
        return const(value)
    elif isinstance(value, string_types):
        return _api_internal._str(value)
    elif isinstance(value, (list, tuple)):
        value = [convert_to_node(x) for x in value]
        return _api_internal._Array(*value)
    elif isinstance(value, dict):
        vlist = []
        for item in value.items():
            if not isinstance(item[0], _CLASS_NODE_BASE):
                raise ValueError("key of map must already been a container type")
            vlist.append(item[0])
            vlist.append(convert_to_node(item[1]))
        return _api_internal._Map(*vlist)
    elif isinstance(value, NodeGeneric):
        return value.asnode()
    else:
        raise ValueError("don't know how to convert type %s to node" % type(value))

def const(value, dtype=None):
    """Construct a constant value for a given type.

    Parameters
    ----------
    value : int or float
        The input value

    dtype : str
        The data type.

    Returns
    -------
    expr : Expr
        Constant expression corresponds to the value.
    """
    if dtype is None:
        if isinstance(value, Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)
