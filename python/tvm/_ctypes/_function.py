# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches
"""Symbolic configuration API."""
from __future__ import absolute_import

import ctypes
import sys
from numbers import Number, Integral

from .._base import _LIB, check_call
from .._base import c_str, py_str, string_types
from ._types import TVMValue, TypeCode, TVMType, TVMByteArray
from ._types import TVMPackedCFunc, TVMCFuncFinalizer
from ._types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH
from ._node import NodeBase, SliceBase, convert_to_node
from ._ndarray import NDArrayBase

FunctionHandle = ctypes.c_void_p
TVMRetValueHandle = ctypes.c_void_p

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
    def cfun(args, type_codes, num_args, ret, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = [C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args)]
        rv = local_pyfunc(*pyargs)
        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one reurn value")
            temp_args = []
            values, tcodes, _ = _make_tvm_args((rv,), temp_args)
            if not isinstance(ret, TVMRetValueHandle):
                ret = TVMRetValueHandle(ret)
            check_call(_LIB.TVMCFuncSetReturn(ret, values[0], ctypes.c_int(tcodes[0])))
            _ = temp_args
            _ = rv

    handle = FunctionHandle()
    f = TVMPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # TVM_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    check_call(_LIB.TVMFuncCreateFromCFunc(
        f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)))
    return Function(handle)


def _make_tvm_args(args, temp_args):
    """Pack arguments into c args tvm call accept"""
    num_args = len(args)
    values = (TVMValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = ctypes.cast(arg.handle, ctypes.c_void_p)
            type_codes[i] = TypeCode.ARRAY_HANDLE
        elif isinstance(arg, NodeBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, TVMType):
            values[i].v_str = c_str(str(arg))
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, bytearray):
            arr = TVMByteArray()
            arr.data = ctypes.cast(
                (ctypes.c_byte * len(arg)).from_buffer(arg),
                ctypes.POINTER(ctypes.c_byte))
            arr.size = len(arg)
            values[i].v_handle = ctypes.c_void_p(ctypes.addressof(arr))
            temp_args.append(arr)
            type_codes[i] = TypeCode.BYTES
        elif isinstance(arg, string_types):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple, dict, SliceBase)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, Function):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
        elif callable(arg):
            arg = convert_to_tvm_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
            temp_args.append(arg)
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class Function(object):
    """A function object at runtime."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global=False):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool, optional
            Whether it is global function
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global:
            check_call(_LIB.TVMFuncFree(self.handle))

    def __call__(self, *args):
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        check_call(_LIB.TVMFuncCall(
            self.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)


def _handle_return_func(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return Function(handle, False)

# setup return handle for function type
RETURN_SWITCH[TypeCode.FUNC_HANDLE] = _handle_return_func

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
        if not isinstance(myf, Function):
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
    return Function(handle, True)


def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMFuncListGlobalNames(ctypes.byref(size),
                                           ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames


def _init_api_functions(root_namespace):
    """List and add all the functions to current module."""
    module_obj = sys.modules["%s.api" % root_namespace]
    module_internal = sys.modules["%s._api_internal" % root_namespace]
    namespace_match = {
        "_make_": sys.modules["%s.make" % root_namespace],
        "_arith_": sys.modules["%s.arith" % root_namespace],
        "_pass_": sys.modules["%s.ir_pass" % root_namespace],
        "_codegen_": sys.modules["%s.codegen" % root_namespace],
        "_schedule_": sys.modules["%s.schedule" % root_namespace]
    }
    for name in list_global_func_names():
        fname = name
        target_module = module_internal if name.startswith('_') else module_obj
        for k, v in namespace_match.items():
            if name.startswith(k):
                fname = name[len(k):]
                target_module = v
        f = get_global_func(name)
        setattr(target_module, fname, f)
