"""Common runtime ctypes."""
# pylint: disable=invalid-name
from __future__ import absolute_import

import ctypes
import numpy as np
from .base import _LIB, check_call
from .. import _api_internal

tvm_shape_index_t = ctypes.c_int64

class TVMByteArray(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)),
                ("size", ctypes.c_size_t)]

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


class TVMContext(ctypes.Structure):
    """TVM context strucure."""
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        4 : 'opencl',
        8 : 'metal',
        9 : 'vpi'
    }
    STR2MASK = {
        'cpu': 1,
        'gpu': 2,
        'cuda': 2,
        'cl': 4,
        'opencl': 4,
        'metal': 8,
        'vpi': 9
    }
    def __init__(self, device_type, device_id):
        super(TVMContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    @property
    def exist(self):
        """Whether this device exist."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 0) != 0

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 1)

    @property
    def warp_size(self):
        """Number of threads that executes in concurrent."""
        return _api_internal._GetDeviceAttr(
            self.device_type, self.device_id, 2)

    def sync(self):
        """Synchronize until jobs finished at the context."""
        check_call(_LIB.TVMSynchronize(self, None))

    def __eq__(self, other):
        return (isinstance(other, TVMContext) and
                self.device_id == other.device_id and
                self.device_type == other.device_type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s(%d)" % (
            TVMContext.MASK2STR[self.device_type], self.device_id)


class TVMArray(ctypes.Structure):
    """TVMValue in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", TVMContext),
                ("ndim", ctypes.c_int),
                ("dtype", TVMType),
                ("shape", ctypes.POINTER(tvm_shape_index_t)),
                ("strides", ctypes.POINTER(tvm_shape_index_t)),
                ("byte_offset", ctypes.c_size_t)]

TVMArrayHandle = ctypes.POINTER(TVMArray)
