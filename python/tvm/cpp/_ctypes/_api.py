# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys
from numbers import Number as Number

from .._base import _LIB
from .._base import c_str, py_str, string_types
from .._base import FunctionHandle, NodeHandle
from .._base import check_call, ctypes2docstring


class ArgVariant(ctypes.Union):
    _fields_ = [("v_long", ctypes.c_long),
                ("v_double", ctypes.c_double),
                ("v_str", ctypes.c_char_p),
                ("v_handle", ctypes.c_void_p)]

kNull = 0
kLong = 1
kDouble = 2
kStr = 3
kNodeHandle = 4

RET_SWITCH = None

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

    def __del__(self):
        check_call(_LIB.TVMNodeFree(self.handle))

    def __getattr__(self, name):
        ret_val = ArgVariant()
        ret_typeid = ctypes.c_int()
        check_call(_LIB.TVMNodeGetAttr(
            self.handle, c_str(name),
            ctypes.byref(ret_val), ctypes.byref(ret_typeid)))
        ret = RET_SWITCH[ret_typeid.value](ret_val)


def _type_key(handle):
    ret_val = ArgVariant()
    ret_typeid = ctypes.c_int()
    check_call(_LIB.TVMNodeGetAttr(
        handle, c_str("type_key"),
        ctypes.byref(ret_val), ctypes.byref(ret_typeid)))
    return py_str(ret_val.v_str)

NODE_TYPE = {
}

RET_SWITCH = {
    kNull: lambda x: None,
    kLong: lambda x: x.v_long.value,
    kDouble: lambda x: x.v_double.value,
    kStr: lambda x: py_str(x.v_str),
    kNodeHandle: lambda x: NODE_TYPE.get(_type_key(x), NodeBase)(x.v_handle)
}

def _push_arg(arg):
    a = ArgVariant()
    if arg is None:
        _LIB.TVMPushStack(a, ctypes.c_int(kNull))
    elif isinstance(arg, NodeBase):
        a.v_handle = arg.handle
        _LIB.TVMPushStack(a, ctypes.c_int(kNodeHandle))
    elif isinstance(arg, int):
        a.v_long = ctypes.c_long(arg)
        _LIB.TVMPushStack(a, ctypes.c_int(kLong))
    elif isinstance(arg, Number):
        a.v_double = ctypes.c_double(arg)
        _LIB.TVMPushStack(a, ctypes.c_int(kDouble))
    elif isinstance(arg, string_types):
        a.v_str = c_str(arg)
        _LIB.TVMPushStack(a, ctypes.c_int(kStr))
    else:
        raise TypeError("Don't know how to handle type %s" % type(arg))


def _make_function(handle, name):
    """Create an atomic symbol function by handle and funciton name."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = ctypes.c_int()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.TVMGetFunctionInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(ret_type)))

    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    func_name = name
    desc = py_str(desc.value)

    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n' +
               '    The result symbol.')
    doc_str = doc_str % (desc, param_str)
    arg_names = [py_str(arg_names[i]) for i in range(num_args.value)]

    def func(*args, **kwargs):
        """TVM function"""
        for arg in args:
            _push_arg(arg)
        ret_val = ArgVariant()
        ret_typeid = ctypes.c_int()
        check_call(_LIB.TVMFunctionCall(
            handle, ctypes.byref(ret_val), ctypes.byref(ret_typeid)))
        return RET_SWITCH[ret_typeid.value](ret_val)

    func.__name__ = func_name
    func.__doc__ = doc_str
    return func


def register_node(type_key):
    """register node type

    Parameters
    ----------
    type_key : str
        The type key of the node
    """
    def register(cls):
        NODE_TYPE[type_key] = cls
    return register


def _init_function_module(root_namespace):
    """List and add all the functions to current module."""
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMListFunctionNames(ctypes.byref(size),
                                         ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_obj = sys.modules["%s.function" % root_namespace]
    module_internal = sys.modules["%s._function_internal" % root_namespace]
    for name in op_names:
        hdl = FunctionHandle()
        check_call(_LIB.TVMGetFunctionHandle(c_str(name), ctypes.byref(hdl)))
        function = _make_function(hdl, name)
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
