"""The C Types used in API."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import ctypes
import numpy as np
from .._base import py_str, check_call, _LIB

tvm_shape_index_t = ctypes.c_int64

class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    ARRAY_HANDLE = 5
    TVM_TYPE = 6
    NODE_HANDLE = 7
    MODULE_HANDLE = 8
    FUNC_HANDLE = 9
    STR = 10
    BYTES = 11

def _api_type(code):
    """create a type accepted by API"""
    t = TVMType()
    t.bits = 64
    t.lanes = 1
    t.type_code = code
    return t


class TVMType(ctypes.Structure):
    """TVM datatype structure"""
    _fields_ = [("type_code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        0 : 'int',
        1 : 'uint',
        2 : 'float',
        4 : 'handle'
    }
    def __init__(self, type_str, lanes=1):
        super(TVMType, self).__init__()
        if isinstance(type_str, np.dtype):
            type_str = str(type_str)
        if type_str.startswith("int"):
            self.type_code = 0
            bits = int(type_str[3:])
        elif type_str.startswith("uint"):
            self.type_code = 1
            bits = int(type_str[4:])
        elif type_str.startswith("float"):
            self.type_code = 2
            bits = int(type_str[5:])
        elif type_str.startswith("handle"):
            self.type_code = 4
            bits = 64
        else:
            raise ValueError("Donot know how to handle type %s" % type_str)

        bits = 32 if bits == 0 else bits
        if (bits & (bits - 1)) != 0 or bits < 8:
            raise ValueError("Donot know how to handle type %s" % type_str)
        self.bits = bits
        self.lanes = lanes

    def __repr__(self):
        x = "%s%d" % (TVMType.CODE2STR[self.type_code], self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (self.bits == other.bits and
                self.type_code == other.type_code and
                self.lanes == other.lanes)

    def __ne__(self, other):
        return not self.__eq__(other)


class TVMValue(ctypes.Union):
    """TVMValue in C API"""
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p)]

class TVMByteArray(ctypes.Structure):
    """TVM datatype structure"""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)),
                ("size", ctypes.c_size_t)]


TVMPackedCFunc = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(TVMValue),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p)


TVMCFuncFinalizer = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p)


def _return_handle(x):
    """return handle"""
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    return handle

def _return_bytes(x):
    """return handle"""
    handle = x.v_handle
    if not isinstance(handle, ctypes.c_void_p):
        handle = ctypes.c_void_p(handle)
    arr = ctypes.cast(handle, ctypes.POINTER(TVMByteArray))[0]
    size = arr.size
    res = bytearray(size)
    rptr = (ctypes.c_byte * size).from_buffer(res)
    if not ctypes.memmove(rptr, arr.data, size):
        raise RuntimeError('memmove failed')
    return res

def _wrap_arg_func(return_f, type_code):
    tcode = ctypes.c_int(type_code)
    def _wrap_func(x):
        check_call(_LIB.TVMCbArgToReturn(ctypes.byref(x), tcode))
        return return_f(x)
    return _wrap_func

RETURN_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.HANDLE: _return_handle,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.BYTES: _return_bytes
}

C_TO_PY_ARG_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.HANDLE: _return_handle,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.BYTES: _return_bytes
}
