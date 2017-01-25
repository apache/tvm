# coding: utf-8
# pylint: disable=invalid-name, protected-access
# pylint: disable=no-member, missing-docstring
"""Symbolic configuration API."""
from __future__ import absolute_import

import ctypes
from numbers import Number, Integral

from .._base import _LIB, check_call
from .._base import c_str, py_str, string_types
from .. import _api_internal
from ._types import TVMValue, TypeCode, RETURN_SWITCH

NodeHandle = ctypes.c_void_p

"""Maps node type to its constructor"""
NODE_TYPE = {
}

def _return_node(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, NodeHandle):
        handle = NodeHandle(handle)
    ret_val = TVMValue()
    ret_type_code = ctypes.c_int()
    ret_success = ctypes.c_int()
    check_call(_LIB.TVMNodeGetAttr(
        handle, c_str("type_key"),
        ctypes.byref(ret_val),
        ctypes.byref(ret_type_code),
        ctypes.byref(ret_success)))
    return NODE_TYPE.get(py_str(ret_val.v_str), NodeBase)(handle)


RETURN_SWITCH[TypeCode.NODE_HANDLE] = _return_node


class SliceBase(object):
    """base class of slice object"""
    pass

class NodeBase(object):
    """Symbol is symbolic graph."""
    __slots__ = ["handle"]
    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

    def __repr__(self):
        return _api_internal._format_str(self)

    def __del__(self):
        check_call(_LIB.TVMNodeFree(self.handle))

    def __getattr__(self, name):
        ret_val = TVMValue()
        ret_type_code = ctypes.c_int()
        ret_success = ctypes.c_int()
        check_call(_LIB.TVMNodeGetAttr(
            self.handle, c_str(name),
            ctypes.byref(ret_val),
            ctypes.byref(ret_type_code),
            ctypes.byref(ret_success)))
        if not ret_success.value:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (str(type(self)), name))
        return RETURN_SWITCH[ret_type_code.value](ret_val)

    def __hash__(self):
        return _api_internal._raw_ptr(self)

    def __eq__(self, other):
        if not isinstance(other, NodeBase):
            return False
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __dir__(self):
        plist = ctypes.POINTER(ctypes.c_char_p)()
        size = ctypes.c_uint()
        check_call(_LIB.TVMNodeListAttrNames(
            self.handle, ctypes.byref(size), ctypes.byref(plist)))
        names = []
        for i in range(size.value):
            names.append(py_str(plist[i]))
        return names

    def __reduce__(self):
        return (type(self), (None,), self.__getstate__())

    def __getstate__(self):
        handle = self.handle
        if handle is not None:
            return {'handle': _api_internal._save_json(self)}
        else:
            return {'handle': None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot
        handle = state['handle']
        if handle is not None:
            json_str = handle
            other = _api_internal._load_json(json_str)
            self.handle = other.handle
            other.handle = None
        else:
            self.handle = None


def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)


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
    if isinstance(value, NodeBase):
        return value
    elif isinstance(value, Number):
        return const(value)
    elif isinstance(value, string_types):
        return _api_internal._str(value)
    elif isinstance(value, (list, tuple)):
        value = [convert_to_node(x) for x in value]
        return _api_internal._Array(*value)
    elif isinstance(value, dict):
        vlist = []
        for it in value.items():
            if not isinstance(it[0], NodeBase):
                raise ValueError("key of map must already been a container type")
            vlist.append(it[0])
            vlist.append(convert_to_node(it[1]))
        return _api_internal._Map(*vlist)
    elif isinstance(value, SliceBase):
        return value.tensor(*value.indices)
    else:
        raise ValueError("don't know how to convert type %s to node" % type(value))


def register_node(type_key=None):
    """register node type

    Parameters
    ----------
    type_key : str or cls
        The type key of the node
    """
    if isinstance(type_key, str):
        def register(cls):
            """internal register function"""
            NODE_TYPE[type_key] = cls
            return cls
        return register
    else:
        cls = type_key
        NODE_TYPE[cls.__name__] = cls
        return cls
