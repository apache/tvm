# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Common runtime ctypes."""
# pylint: disable=invalid-name
import ctypes
import json
import numpy as np
from .base import _LIB, check_call

tvm_shape_index_t = ctypes.c_int64

class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    TVM_TYPE = 5
    TVM_CONTEXT = 6
    DLTENSOR_HANDLE = 7
    OBJECT_HANDLE = 8
    MODULE_HANDLE = 9
    PACKED_FUNC_HANDLE = 10
    STR = 11
    BYTES = 12
    NDARRAY_HANDLE = 13
    OBJECT_RVALUE_REF_ARG = 14
    EXT_BEGIN = 15


class TVMByteArray(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)),
                ("size", ctypes.c_size_t)]


class DataType(ctypes.Structure):
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
    def __init__(self, type_str):
        super(DataType, self).__init__()
        if isinstance(type_str, np.dtype):
            type_str = str(type_str)

        if type_str == "bool":
            self.bits = 1
            self.type_code = 1
            self.lanes = 1
            return

        arr = type_str.split("x")
        head = arr[0]
        self.lanes = int(arr[1]) if len(arr) > 1 else 1
        bits = 32

        if head.startswith("int"):
            self.type_code = 0
            head = head[3:]
        elif head.startswith("uint"):
            self.type_code = 1
            head = head[4:]
        elif head.startswith("float"):
            self.type_code = 2
            head = head[5:]
        elif head.startswith("handle"):
            self.type_code = 4
            bits = 64
            head = ""
        elif head.startswith("custom"):
            # pylint: disable=import-outside-toplevel
            import tvm.runtime._ffi_api
            low, high = head.find('['), head.find(']')
            if not low or not high or low >= high:
                raise ValueError("Badly formatted custom type string %s" % type_str)
            type_name = head[low + 1:high]
            self.type_code = tvm.runtime._ffi_api._datatype_get_type_code(type_name)
            head = head[high+1:]
        else:
            raise ValueError("Do not know how to handle type %s" % type_str)
        bits = int(head) if head else bits
        self.bits = bits


    def __repr__(self):
        # pylint: disable=import-outside-toplevel
        if self.bits == 1 and self.lanes == 1:
            return "bool"
        if self.type_code in DataType.CODE2STR:
            type_name = DataType.CODE2STR[self.type_code]
        else:
            import tvm.runtime._ffi_api
            type_name = "custom[%s]" % \
                        tvm.runtime._ffi_api._datatype_get_type_name(self.type_code)
        x = "%s%d" % (type_name, self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (self.bits == other.bits and
                self.type_code == other.type_code and
                self.lanes == other.lanes)

    def __ne__(self, other):
        return not self.__eq__(other)

RPC_SESS_MASK = 128

class TVMContext(ctypes.Structure):
    """TVM context strucure."""
    _fields_ = [("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        4 : 'opencl',
        5 : 'aocl',
        6 : 'sdaccel',
        7 : 'vulkan',
        8 : 'metal',
        9 : 'vpi',
        10: 'rocm',
        11: 'opengl',
        12: 'ext_dev',
        13: 'micro_dev',
        14: 'hexagon',
    }
    STR2MASK = {
        'llvm': 1,
        'stackvm': 1,
        'cpu': 1,
        'c': 1,
        'gpu': 2,
        'cuda': 2,
        'nvptx': 2,
        'cl': 4,
        'opencl': 4,
        'aocl' : 5,
        'aocl_sw_emu' : 5,
        'sdaccel': 6,
        'vulkan': 7,
        'metal': 8,
        'vpi': 9,
        'rocm': 10,
        'opengl': 11,
        'ext_dev': 12,
        'micro_dev': 13,
        'hexagon': 14,
    }
    def __init__(self, device_type, device_id):
        super(TVMContext, self).__init__()
        self.device_type = device_type
        self.device_id = device_id

    def _GetDeviceAttr(self, device_type, device_id, attr_id):
        """Internal helper function to invoke runtime.GetDeviceAttr"""
        # pylint: disable=import-outside-toplevel
        import tvm.runtime._ffi_api
        return tvm.runtime._ffi_api.GetDeviceAttr(
            device_type, device_id, attr_id)

    @property
    def exist(self):
        """Whether this device exist."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 0) != 0

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 1)

    @property
    def warp_size(self):
        """Number of threads that executes in concurrent."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 2)

    @property
    def max_shared_memory_per_block(self):
        """Total amount of shared memory per block in bytes."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 3)

    @property
    def compute_version(self):
        """Get compute verison number in string.

        Currently used to get compute capability of CUDA device.

        Returns
        -------
        version : str
            The version string in `major.minor` format.
        """
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 4)

    @property
    def device_name(self):
        """Return the string name of device."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 5)

    @property
    def max_clock_rate(self):
        """Return the max clock frequency of device."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 6)

    @property
    def multi_processor_count(self):
        """Return the number of compute units of device."""
        return self._GetDeviceAttr(
            self.device_type, self.device_id, 7)

    @property
    def max_thread_dimensions(self):
        """Return the maximum size of each thread axis

        Returns
        -------
        dims: List of int
            The maximum length of threadIdx.x, threadIdx.y, threadIdx.z
        """
        return json.loads(self._GetDeviceAttr(
            self.device_type, self.device_id, 8))

    def sync(self):
        """Synchronize until jobs finished at the context."""
        check_call(_LIB.TVMSynchronize(self.device_type, self.device_id, None))

    def __eq__(self, other):
        return (isinstance(other, TVMContext) and
                self.device_id == other.device_id and
                self.device_type == other.device_type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if self.device_type >= RPC_SESS_MASK:
            tbl_id = self.device_type / RPC_SESS_MASK - 1
            dev_type = self.device_type % RPC_SESS_MASK
            return "remote[%d]:%s(%d)" % (
                tbl_id, TVMContext.MASK2STR[dev_type], self.device_id)
        return "%s(%d)" % (
            TVMContext.MASK2STR[self.device_type], self.device_id)


class TVMArray(ctypes.Structure):
    """TVMValue in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", TVMContext),
                ("ndim", ctypes.c_int),
                ("dtype", DataType),
                ("shape", ctypes.POINTER(tvm_shape_index_t)),
                ("strides", ctypes.POINTER(tvm_shape_index_t)),
                ("byte_offset", ctypes.c_uint64)]


class ObjectRValueRef:
    """Represent an RValue ref to an object that can be moved.

    Parameters
    ----------
    obj : tvm.runtime.Object
        The object that this value refers to
    """
    __slots__ = ["obj"]
    def __init__(self, obj):
        self.obj = obj


TVMArrayHandle = ctypes.POINTER(TVMArray)
