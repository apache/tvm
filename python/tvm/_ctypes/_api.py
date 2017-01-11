# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys
from numbers import Number, Integral

from .._base import _LIB
from .._base import c_str, py_str, string_types
from .._base import FunctionHandle, NodeHandle
from .._base import check_call, ctypes2docstring
from .. import _function_internal

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

NODE_TYPE = {
}

def _return_node(x):
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    ret_val = ArgVariant()
    ret_typeid = ctypes.c_int()
    ret_success = ctypes.c_int()
    check_call(_LIB.TVMNodeGetAttr(
        handle, c_str("type_key"),
        ctypes.byref(ret_val),
        ctypes.byref(ret_typeid),
        ctypes.byref(ret_success)))
    return NODE_TYPE.get(py_str(ret_val.v_str), NodeBase)(handle)

RET_SWITCH = {
    kNull: lambda x: None,
    kLong: lambda x: x.v_long,
    kDouble: lambda x: x.v_double,
    kStr: lambda x: py_str(x.v_str),
    kNodeHandle: lambda x: _return_node(x)
}

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
        return _function_internal._format_str(self)

    def __del__(self):
        check_call(_LIB.TVMNodeFree(self.handle))

    def __getattr__(self, name):
        ret_val = ArgVariant()
        ret_typeid = ctypes.c_int()
        ret_success = ctypes.c_int()
        check_call(_LIB.TVMNodeGetAttr(
            self.handle, c_str(name),
            ctypes.byref(ret_val),
            ctypes.byref(ret_typeid),
            ctypes.byref(ret_success)))
        value = RET_SWITCH[ret_typeid.value](ret_val)
        if not ret_success.value:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (str(type(self)), name))
        return value


    def __hash__(self):
        return _function_internal._raw_ptr(self)

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

def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _function_internal._const(value, dtype)


def convert(value):
    """Convert a value to expression."""
    if isinstance(value, Number):
        return const(value)
    elif isinstance(value, (list, tuple)):
        value = [convert(x) for x in value]
        return _function_internal._Array(*value)
    elif isinstance(value, dict):
        vlist = []
        for it in value.items():
            if not isinstance(it[0], NodeBase):
                raise ValueError("key of map must already been a container type")
            vlist.append(it[0])
            vlist.append(convert(it[1]))
        return _function_internal._Map(*vlist)
    elif isinstance(value, SliceBase):
        return value.tensor(*value.indices)
    else:
        if not isinstance(value, NodeBase):
            raise ValueError("don't know how to handle type %s" % type(value))
        return value


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

    def func(*args):
        """TVM function"""
        cargs = []
        for x in args:
            if isinstance(x, (list, tuple, SliceBase)):
                cargs.append(convert(x))
            else:
                cargs.append(x)

        for arg in cargs:
            _push_arg(arg)
        ret_val = ArgVariant()
        ret_typeid = ctypes.c_int()
        check_call(_LIB.TVMFunctionCall(
            handle, ctypes.byref(ret_val), ctypes.byref(ret_typeid)))
        return RET_SWITCH[ret_typeid.value](ret_val)

    func.__name__ = func_name
    func.__doc__ = doc_str
    return func


def register_node(type_key=None):
    """register node type

    Parameters
    ----------
    type_key : str or cls
        The type key of the node
    """
    if isinstance(type_key, str):
        def register(cls):
            NODE_TYPE[type_key] = cls
            return cls
        return register
    else:
        cls = type_key
        NODE_TYPE[cls.__name__] = cls
        return cls

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
    namespace_match = {
        "_make_" : sys.modules["%s.make" % root_namespace],
        "_pass_" : sys.modules["%s.ir_pass" % root_namespace],
        "_schedule_" : sys.modules["%s.schedule" % root_namespace]
    }

    for name in op_names:
        hdl = FunctionHandle()
        check_call(_LIB.TVMGetFunctionHandle(c_str(name), ctypes.byref(hdl)))
        fname = name
        target_module = module_internal if name.startswith('_') else module_obj
        for k, v in namespace_match.items():
            if name.startswith(k):
                fname = name[len(k):]
                target_module = v
        function = _make_function(hdl, fname)
        setattr(target_module, function.__name__, function)
