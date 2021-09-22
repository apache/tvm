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


class ArgTypeCode(object):
    """Type code used in API calls"""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    TVM_TYPE = 5
    DLDEVICE = 6
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

    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)), ("size", ctypes.c_size_t)]


class DataTypeCode(object):
    """DataType code in DLTensor."""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    BFLOAT = 4


class DataType(ctypes.Structure):
    """TVM datatype structure"""

    _fields_ = [("type_code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        DataTypeCode.INT: "int",
        DataTypeCode.UINT: "uint",
        DataTypeCode.FLOAT: "float",
        DataTypeCode.HANDLE: "handle",
        DataTypeCode.BFLOAT: "bfloat",
    }
    NUMPY2STR = {
        np.dtype(np.bool_): "bool",
        np.dtype(np.int8): "int8",
        np.dtype(np.int16): "int16",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.uint64): "uint64",
        np.dtype(np.float16): "float16",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
        np.dtype(np.float_): "float64",
    }
    STR2DTYPE = {
        "bool": {"type_code": DataTypeCode.UINT, "bits": 1, "lanes": 1},
        "int8": {"type_code": DataTypeCode.INT, "bits": 8, "lanes": 1},
        "int16": {"type_code": DataTypeCode.INT, "bits": 16, "lanes": 1},
        "int32": {"type_code": DataTypeCode.INT, "bits": 32, "lanes": 1},
        "int64": {"type_code": DataTypeCode.INT, "bits": 64, "lanes": 1},
        "uint8": {"type_code": DataTypeCode.UINT, "bits": 8, "lanes": 1},
        "uint16": {"type_code": DataTypeCode.UINT, "bits": 16, "lanes": 1},
        "uint32": {"type_code": DataTypeCode.UINT, "bits": 32, "lanes": 1},
        "uint64": {"type_code": DataTypeCode.UINT, "bits": 64, "lanes": 1},
        "float16": {"type_code": DataTypeCode.FLOAT, "bits": 16, "lanes": 1},
        "float32": {"type_code": DataTypeCode.FLOAT, "bits": 32, "lanes": 1},
        "float64": {"type_code": DataTypeCode.FLOAT, "bits": 64, "lanes": 1},
    }

    def __init__(self, type_str):
        super(DataType, self).__init__()
        numpy_str_map = DataType.NUMPY2STR
        if type_str in numpy_str_map:
            type_str = numpy_str_map[type_str]
        elif isinstance(type_str, np.dtype):
            type_str = str(type_str)

        assert isinstance(type_str, str)

        str_dtype_map = DataType.STR2DTYPE
        if type_str in str_dtype_map:
            dtype_map = str_dtype_map[type_str]
            self.bits = dtype_map["bits"]
            self.type_code = dtype_map["type_code"]
            self.lanes = dtype_map["lanes"]
            return

        arr = type_str.split("x")
        head = arr[0]
        self.lanes = int(arr[1]) if len(arr) > 1 else 1
        bits = 32

        if head.startswith("int"):
            self.type_code = DataTypeCode.INT
            head = head[3:]
        elif head.startswith("uint"):
            self.type_code = DataTypeCode.UINT
            head = head[4:]
        elif head.startswith("float"):
            self.type_code = DataTypeCode.FLOAT
            head = head[5:]
        elif head.startswith("handle"):
            self.type_code = DataTypeCode.HANDLE
            bits = 64
            head = ""
        elif head.startswith("bfloat"):
            self.type_code = DataTypeCode.BFLOAT
            head = head[6:]
        elif head.startswith("custom"):
            # pylint: disable=import-outside-toplevel
            import tvm.runtime._ffi_api

            low, high = head.find("["), head.find("]")
            if not low or not high or low >= high:
                raise ValueError("Badly formatted custom type string %s" % type_str)
            type_name = head[low + 1 : high]
            self.type_code = tvm.runtime._ffi_api._datatype_get_type_code(type_name)
            head = head[high + 1 :]
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

            type_name = "custom[%s]" % tvm.runtime._ffi_api._datatype_get_type_name(self.type_code)
        x = "%s%d" % (type_name, self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (
            self.bits == other.bits
            and self.type_code == other.type_code
            and self.lanes == other.lanes
        )

    def __ne__(self, other):
        return not self.__eq__(other)


RPC_SESS_MASK = 128


class Device(ctypes.Structure):
    """TVM device strucure.

    Typically constructed using convenience function
    :meth:`tvm.runtime.device`.

    Exposes uniform interface to device-specific APIs such as CUDA or
    OpenCL.  Some properties may return None depending on whether an
    API exposes that particular property.

    """

    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]
    MASK2STR = {
        1: "cpu",
        2: "cuda",
        4: "opencl",
        5: "aocl",
        6: "sdaccel",
        7: "vulkan",
        8: "metal",
        9: "vpi",
        10: "rocm",
        12: "ext_dev",
        14: "hexagon",
        15: "webgpu",
    }
    STR2MASK = {
        "llvm": 1,
        "stackvm": 1,
        "cpu": 1,
        "c": 1,
        "cuda": 2,
        "nvptx": 2,
        "cl": 4,
        "opencl": 4,
        "aocl": 5,
        "aocl_sw_emu": 5,
        "sdaccel": 6,
        "vulkan": 7,
        "metal": 8,
        "vpi": 9,
        "rocm": 10,
        "ext_dev": 12,
        "hexagon": 14,
        "webgpu": 15,
    }

    def __init__(self, device_type, device_id):
        super(Device, self).__init__()
        self.device_type = int(device_type)
        self.device_id = device_id

    def _GetDeviceAttr(self, device_type, device_id, attr_id):
        """Internal helper function to invoke runtime.GetDeviceAttr"""
        # pylint: disable=import-outside-toplevel
        import tvm.runtime._ffi_api

        return tvm.runtime._ffi_api.GetDeviceAttr(device_type, device_id, attr_id)

    @property
    def exist(self):
        """Whether this device exists.

        Returns True if TVM has support for the device, if the
        physical device is present, and the device is accessible
        through appropriate drivers (e.g. cuda/vulkan).

        Returns
        -------
        exist : bool
            True if the device exists

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 0) != 0

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block.

        Returns device value for cuda, metal, rocm, opencl, and vulkan
        devices.  Returns remote device value for RPC devices.
        Returns None for all other devices.

        Returns
        -------
        max_threads_per_block : int or None
            The number of threads on each block

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 1)

    @property
    def warp_size(self):
        """Number of threads that execute concurrently.

        Returns device value for for cuda, rocm, and vulkan.  Returns
        1 for metal and opencl devices, regardless of the physical
        device.  Returns remote device value for RPC devices.  Returns
        None for all other devices.

        Returns
        -------
        warp_size : int or None
            Number of threads that execute concurrently

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 2)

    @property
    def max_shared_memory_per_block(self):
        """Total amount of shared memory per block in bytes.

        Returns device value for cuda, rocm, opencl, and vulkan.
        Returns remote device value for RPC devices.  Returns None for
        all other devices.

        Returns
        -------
        max_shared_memory_per_block : int or None
            Total amount of shared memory per block in bytes

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 3)

    @property
    def compute_version(self):
        """Get compute version number as string.

        Returns maximum API version (e.g. CUDA/OpenCL/Vulkan)
        supported by the device.

        Returns device value for cuda, rocm, opencl, and
        vulkan. Returns remote device value for RPC devices.  Returns
        None for all other devices.

        Returns
        -------
        version : str or None
            The version string in `major.minor` format.

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 4)

    @property
    def device_name(self):
        """Return the vendor-specific name of device.

        Returns device value for cuda, rocm, opencl, and vulkan.
        Returns remote device value for RPC devices.  Returns None for
        all other devices.

        Returns
        -------
        device_name : str or None
            The name of the device.

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 5)

    @property
    def max_clock_rate(self):
        """Return the max clock frequency of device (kHz).

        Returns device value for cuda, rocm, and opencl.  Returns
        remote device value for RPC devices.  Returns None for all
        other devices.

        Returns
        -------
        max_clock_rate : int or None
            The maximum clock frequency of the device (kHz)

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 6)

    @property
    def multi_processor_count(self):
        """Return the number of compute units in the device.

        Returns device value for cuda, rocm, and opencl.  Returns
        remote device value for RPC devices.  Returns None for all
        other devices.

        Returns
        -------
        multi_processor_count : int or None
            Thee number of compute units in the device

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 7)

    @property
    def max_thread_dimensions(self):
        """Return the maximum size of each thread axis

        Returns device value for cuda, rocm, opencl, and vulkan.
        Returns remote device value for RPC devices.  Returns None for
        all other devices.

        Returns
        -------
        dims: List of int, or None
            The maximum length of threadIdx.x, threadIdx.y, threadIdx.z

        """
        return json.loads(self._GetDeviceAttr(self.device_type, self.device_id, 8))

    @property
    def api_version(self):
        """Returns version number of the SDK used to compile TVM.

        For example, CUDA_VERSION for cuda or VK_HEADER_VERSION for
        Vulkan.

        Returns device value for cuda, rocm, opencl, and vulkan.
        Returns remote device value for RPC devices.  Returns None for
        all other devices.

        Returns
        -------
        version : int or None
            The version of the SDK

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 11)

    @property
    def driver_version(self):
        """Returns version number of the driver

        Returns driver vendor's internal version number.
        (e.g. "450.408.256" for nvidia-driver-450)

        Returns device value for opencl and vulkan.  Returns remote
        device value for RPC devices.  Returns None for all other
        devices.

        Returns
        -------
        version : str or None
            The version string in `major.minor.patch` format.

        """
        return self._GetDeviceAttr(self.device_type, self.device_id, 12)

    def create_raw_stream(self):
        """Create a new runtime stream at the context.

        User should free the stream after use.

        Returns
        -------
        stream : TVMStreamHandle
            The created runtime stream.
        """
        stream = ctypes.c_void_p()
        check_call(_LIB.TVMStreamCreate(self.device_type, self.device_id, ctypes.byref(stream)))
        return stream

    def free_raw_stream(self, stream):
        """Free a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be released.
        """
        check_call(_LIB.TVMStreamFree(self.device_type, self.device_id, stream))

    def set_raw_stream(self, stream):
        """Set a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be set to the device.
        """
        check_call(_LIB.TVMSetStream(self.device_type, self.device_id, stream))

    def sync(self, stream=None):
        """Synchronize until jobs finished at the context.

        Parameters
        ----------
        stream : TVMStreamHandle
            Jobs in this stream should be finished.
        """
        check_call(_LIB.TVMSynchronize(self.device_type, self.device_id, stream))

    def __eq__(self, other):
        return (
            isinstance(other, Device)
            and self.device_id == other.device_id
            and self.device_type == other.device_type
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if self.device_type >= RPC_SESS_MASK:
            tbl_id = self.device_type / RPC_SESS_MASK - 1
            dev_type = self.device_type % RPC_SESS_MASK
            return "remote[%d]:%s(%d)" % (tbl_id, Device.MASK2STR[dev_type], self.device_id)
        return "%s(%d)" % (Device.MASK2STR[self.device_type], self.device_id)


class TVMArray(ctypes.Structure):
    """TVMValue in C API"""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", Device),
        ("ndim", ctypes.c_int),
        ("dtype", DataType),
        ("shape", ctypes.POINTER(tvm_shape_index_t)),
        ("strides", ctypes.POINTER(tvm_shape_index_t)),
        ("byte_offset", ctypes.c_uint64),
    ]


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
