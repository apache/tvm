# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement
"""Function configuration API."""
from __future__ import absolute_import

import ctypes
import traceback
from numbers import Number, Integral

from ..base import _LIB, check_call
from ..base import c_str, string_types
from ..node_generic import convert_to_node, NodeGeneric
from ..runtime_ctypes import TVMType, TVMByteArray, TVMContext
from . import ndarray as _nd
from .ndarray import NDArrayBase, _make_array
from .types import TVMValue, TypeCode
from .types import TVMPackedCFunc, TVMCFuncFinalizer
from .types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func
from .node import NodeBase

FunctionHandle = ctypes.c_void_p
ModuleHandle = ctypes.c_void_p
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
        pyargs = (C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args))
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception:
            msg = traceback.format_exc()
            _LIB.TVMAPISetLastError(c_str(msg))
            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one return value")
            temp_args = []
            values, tcodes, _ = _make_tvm_args((rv,), temp_args)
            if not isinstance(ret, TVMRetValueHandle):
                ret = TVMRetValueHandle(ret)
            check_call(_LIB.TVMCFuncSetReturn(ret, values, tcodes, ctypes.c_int(1)))
            _ = temp_args
            _ = rv
        return 0

    handle = FunctionHandle()
    f = TVMPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # TVM_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    check_call(_LIB.TVMFuncCreateFromCFunc(
        f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)))
    return _CLASS_FUNCTION(handle, False)


def _make_tvm_args(args, temp_args):
    """Pack arguments into c args tvm call accept"""
    num_args = len(args)
    values = (TVMValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if isinstance(arg, NodeBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
        elif arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = ctypes.cast(arg.handle, ctypes.c_void_p)
            type_codes[i] = TypeCode.ARRAY_HANDLE
        elif isinstance(arg, _nd._TVM_COMPATS):
            values[i].v_handle = ctypes.c_void_p(arg._tvm_handle)
            type_codes[i] = arg.__class__._tvm_tcode
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, TVMType):
            values[i].v_str = c_str(str(arg))
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, TVMContext):
            values[i].v_ctx = arg
            type_codes[i] = TypeCode.TVM_CONTEXT
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
        elif isinstance(arg, (list, tuple, dict, NodeGeneric)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, _CLASS_MODULE):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.MODULE_HANDLE
        elif isinstance(arg, FunctionBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
        elif isinstance(arg, ctypes.c_void_p):
            values[i].v_handle = arg
            type_codes[i] = TypeCode.HANDLE
        elif callable(arg):
            arg = convert_to_tvm_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
            temp_args.append(arg)
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global:
            check_call(_LIB.TVMFuncFree(self.handle))

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
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

def _return_module(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, ModuleHandle):
        handle = ModuleHandle(handle)
    return _CLASS_MODULE(handle)

def _handle_return_func(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return _CLASS_FUNCTION(handle, False)


# setup return handle for function type
RETURN_SWITCH[TypeCode.FUNC_HANDLE] = _handle_return_func
RETURN_SWITCH[TypeCode.MODULE_HANDLE] = _return_module
C_TO_PY_ARG_SWITCH[TypeCode.FUNC_HANDLE] = _wrap_arg_func(
    _handle_return_func, TypeCode.FUNC_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.MODULE_HANDLE] = _wrap_arg_func(
    _return_module, TypeCode.MODULE_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.ARRAY_HANDLE] = lambda x: _make_array(x.v_handle, True)

_CLASS_MODULE = None
_CLASS_FUNCTION = None

def _set_class_module(module_class):
    """Initialize the module."""
    global _CLASS_MODULE
    _CLASS_MODULE = module_class

def _set_class_function(func_class):
    global _CLASS_FUNCTION
    _CLASS_FUNCTION = func_class
