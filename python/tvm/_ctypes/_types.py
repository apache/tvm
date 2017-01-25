"""The C Types used in API."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import ctypes
import numpy as np

class TVMValue(ctypes.Union):
    """TVMValue in C API"""
    _fields_ = [("v_int64", ctypes.c_int64),
                ("v_float64", ctypes.c_double),
                ("v_handle", ctypes.c_void_p),
                ("v_str", ctypes.c_char_p)]

class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    NODE_HANDLE = 5
    STR = 6
    FUNC_HANDLE = 7

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
        2 : 'float'
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


TVMPackedCFunc = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(TVMValue),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_void_p)


TVMCFuncFinalizer = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p)
