# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
# pylint: disable=attribute-defined-outside-init, no-member, missing-docstring, too-many-return-statements
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys
from numbers import Number, Integral

from .._base import _LIB
from .._base import c_str, py_str, string_types
from .._base import check_call, ctypes2docstring
from .. import _api_internal
from . import _runtime_api
from ._types import TVMValue, TypeCode, TVMPackedCFunc, TVMCFuncFinalizer

# type definitions
APIFuncHandle = ctypes.c_void_p
NodeHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p

class APIType(object):
    """TVMType used in API calls"""
    INT = ctypes.c_int(TypeCode.INT)
    UINT = ctypes.c_int(TypeCode.UINT)
    FLOAT = ctypes.c_int(TypeCode.FLOAT)
    HANDLE = ctypes.c_int(TypeCode.HANDLE)
    NULL = ctypes.c_int(TypeCode.NULL)
    NODE_HANDLE = ctypes.c_int(TypeCode.NODE_HANDLE)
    STR = ctypes.c_int(TypeCode.STR)
    FUNC_HANDLE = ctypes.c_int(TypeCode.FUNC_HANDLE)


NODE_TYPE = {
}

def _return_node(x):
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


def _return_func(x):
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return _runtime_api._function_cls(handle)


def _return_handle(x):
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    return handle


RET_SWITCH = {
    TypeCode.NULL: lambda x: None,
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.NODE_HANDLE: _return_node,
    TypeCode.FUNC_HANDLE: _return_func
}

PACK_ARG_SWITCH = {
    TypeCode.NULL: lambda x: None,
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.HANDLE: lambda x: _return_handle,
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
        value = RET_SWITCH[ret_type_code.value](ret_val)
        if not ret_success.value:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (str(type(self)), name))
        return value

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
            _push_arg(json_str)
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


def _ctypes_free_resource(rhandle):
    """callback to free resources when it it not needed."""
    pyobj = ctypes.cast(rhandle, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)

# Global callback that is always alive
TVM_FREE_PYOBJ = TVMCFuncFinalizer(_ctypes_free_resource)
ctypes.pythonapi.Py_IncRef(ctypes.py_object(TVM_FREE_PYOBJ))

def convert_to_tvm_func(pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.nd.Function
        The converted tvm function.
    """
    local_pyfunc = pyfunc
    def cfun(args, type_codes, num_args, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = [PACK_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args)]
        local_pyfunc(*pyargs)
    handle = FunctionHandle()
    f = TVMPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # TVM_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    check_call(_LIB.TVMFuncCreateFromCFunc(
        f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)))
    return _runtime_api._function_cls(handle)


def convert(value):
    """Convert a value to expression."""
    if isinstance(value, (NodeBase, _runtime_api.FunctionBase)):
        return value
    elif isinstance(value, Number):
        return const(value)
    elif isinstance(value, string_types):
        return _api_internal._str(value)
    elif isinstance(value, (list, tuple)):
        value = [convert(x) for x in value]
        return _api_internal._Array(*value)
    elif isinstance(value, dict):
        vlist = []
        for it in value.items():
            if not isinstance(it[0], NodeBase):
                raise ValueError("key of map must already been a container type")
            vlist.append(it[0])
            vlist.append(convert(it[1]))
        return _api_internal._Map(*vlist)
    elif isinstance(value, SliceBase):
        return value.tensor(*value.indices)
    elif callable(value):
        return convert_to_tvm_func(value)
    else:
        raise ValueError("don't know how to handle type %s" % type(value))
    return value


def _push_arg(arg):
    a = TVMValue()
    if arg is None:
        _LIB.TVMAPIPushStack(a, APIType.NULL)
    elif isinstance(arg, NodeBase):
        a.v_handle = arg.handle
        _LIB.TVMAPIPushStack(a, APIType.NODE_HANDLE)
    elif isinstance(arg, Integral):
        a.v_int64 = ctypes.c_int64(arg)
        _LIB.TVMAPIPushStack(a, APIType.INT)
    elif isinstance(arg, Number):
        a.v_double = ctypes.c_double(arg)
        _LIB.TVMAPIPushStack(a, APIType.FLOAT)
    elif isinstance(arg, string_types):
        a.v_str = c_str(arg)
        _LIB.TVMAPIPushStack(a, APIType.STR)
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

    check_call(_LIB.TVMGetAPIFuncInfo(
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
               '%s\n')
    doc_str = doc_str % (desc, param_str)
    arg_names = [py_str(arg_names[i]) for i in range(num_args.value)]

    def func(*args):
        """TVM function"""
        cargs = []
        for x in args:
            if isinstance(x, (list, tuple, dict, SliceBase)):
                cargs.append(convert(x))
            else:
                cargs.append(x)

        for arg in cargs:
            _push_arg(arg)
        ret_val = TVMValue()
        ret_type_code = ctypes.c_int()
        check_call(_LIB.TVMAPIFuncCall(
            handle, ctypes.byref(ret_val), ctypes.byref(ret_type_code)))
        return RET_SWITCH[ret_type_code.value](ret_val)

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
            """internal register function"""
            NODE_TYPE[type_key] = cls
            return cls
        return register
    else:
        cls = type_key
        NODE_TYPE[cls.__name__] = cls
        return cls


def register_func(func_name, f=None):
    """Register global function

    Parameters
    ----------
    func_name : str or function
        The function name

    f : function
        The function to be registered.

    Returns
    -------
    fregister : function
        Register function if f is not specified.
    """
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")
    def register(myf):
        """internal register function"""
        if not isinstance(myf, _runtime_api.FunctionBase):
            myf = convert_to_tvm_func(myf)
        check_call(_LIB.TVMFuncRegisterGlobal(
            c_str(func_name), myf.handle))
    if f:
        register(f)
    else:
        return register


def get_global_func(name):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    Returns
    -------
    func : tvm.nd.Function
        The function to be returned.
    """
    handle = FunctionHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    return _runtime_api._function_cls(handle)


def _init_api_module(root_namespace):
    """List and add all the functions to current module."""
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMListAPIFuncNames(ctypes.byref(size),
                                        ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_obj = sys.modules["%s.api" % root_namespace]
    module_internal = sys.modules["%s._api_internal" % root_namespace]
    namespace_match = {
        "_make_": sys.modules["%s.make" % root_namespace],
        "_pass_": sys.modules["%s.ir_pass" % root_namespace],
        "_codegen_": sys.modules["%s.codegen" % root_namespace],
        "_schedule_": sys.modules["%s.schedule" % root_namespace]
    }

    for name in op_names:
        hdl = APIFuncHandle()
        check_call(_LIB.TVMGetAPIFuncHandle(c_str(name), ctypes.byref(hdl)))
        fname = name
        target_module = module_internal if name.startswith('_') else module_obj
        for k, v in namespace_match.items():
            if name.startswith(k):
                fname = name[len(k):]
                target_module = v
        function = _make_function(hdl, fname)
        setattr(target_module, function.__name__, function)
