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
import enum
import functools
import json
import numpy as np
from typing import Union, Optional

from .base import _LIB, check_call
import tvm

tvm_shape_index_t = ctypes.c_int64


class ArgTypeCode(enum.Enum):
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


class DataTypeCode(enum.Enum):
    """DataType code in DLTensor."""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    BFLOAT = 4


class DataType(ctypes.Structure):
    """TVM datatype structure"""

    _fields_ = [
        ("_type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]
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

    def __init__(
        self,
        value: Union[int, str, np.dtype, "DataType"],
        bits: Optional[int] = None,
        lanes: Optional[int] = None,
    ):
        super(DataType, self).__init__()

        if isinstance(value, (str, np.dtype)):
            value = self._unpack_str(value)

        if isinstance(value, DataType):
            assert bits is None
            assert lanes is None
            self.type_code = value.type_code
            self.bits = value.bits
            self.lanes = value.lanes
        else:
            assert bits is not None
            assert lanes is not None
            self.type_code = value
            self.bits = bits
            self.lanes = lanes

    @property
    def type_code(self) -> DataTypeCode:
        """The type code of the datatype

        This internal field must be a `ctypes.c_uint8` to match the
        struct definition.  This wrapper allows the Python API to
        present the `enum.Enum` subclass.
        """
        return DataTypeCode(self._type_code)

    @type_code.setter
    def type_code(self, val: Union[DataTypeCode, int]):
        # Round trip through DataTypeCode ensures that the integer
        # provided is valid.
        if isinstance(val, int):
            val = DataTypeCode(val)

        self._type_code = val.value

    def with_lanes(self, lanes: int) -> "DataType":
        """Return the current datatype with the specified lanes"""
        return DataType(self.type_code, self.bits, lanes)

    @classmethod
    def _ffi_string_to_data_type_func(cls):
        func = getattr(cls, "_string_to_data_type_func", None)
        if func:
            return func

        import tvm  # pylint: disable=import-outside-toplevel

        cls._string_to_data_type = func = tvm._ffi.registry.get_global_func(
            "runtime.String2DLDataType"
        )
        return func

    @classmethod
    def _ffi_data_type_to_string_func(cls):
        func = getattr(cls, "_data_type_to_string_func", None)
        if func:
            return func

        import tvm  # pylint: disable=import-outside-toplevel

        cls._data_type_to_string = func = tvm._ffi.registry.get_global_func(
            "runtime.DLDataType2String"
        )
        return func

    @classmethod
    def _unpack_str(cls, type_str):
        numpy_str_map = DataType.NUMPY2STR
        if type_str in numpy_str_map:
            type_str = numpy_str_map[type_str]
        elif isinstance(type_str, np.dtype):
            type_str = str(type_str)

        assert isinstance(type_str, str)

        return cls._ffi_string_to_data_type_func()(type_str)

    def __str__(self):
        return self._ffi_data_type_to_string_func()(self)

    def __repr__(self):
        return f'DataType("{str(self)}")'

    def __eq__(self, other):
        if isinstance(other, (str, np.dtype)):
            other = DataType(other)

        return (
            self.bits == other.bits
            and self.type_code == other.type_code
            and self.lanes == other.lanes
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (self.type_code, self.bits, self.lanes).__hash__()

    def __contains__(self, search):
        """Backwards compatibility wrapper

        To support use of the datatype as a string.  Use should be
        avoided in the future.

        Example
        -------

        .. code-block:: python

            # Old method, supported by this wrapper
            is_floating_point = "float" in dtype

            # New method, preferred
            is_floating_point = dtype.type_code == DataTypeCode.FLOAT
        """
        return search in str(self)

    def __getitem__(self, index):
        """Backwards compatibility wrapper

        To support use of the datatype as a string.  Use should be
        avoided in the future.

        Example
        -------

        .. code-block:: python

            # Old method, supported by this wrapper
            bits = int(dtype[-2:])

            # New method, preferred
            bits = dtype.bits
        """
        return str(self)[index]

    @property
    def dtype(self):
        """Converter attribute to allow use as a np.dtype

        See https://numpy.org/doc/stable/reference/arrays.dtypes.html,
        under section "Types with .dtype"
        """
        return str(self)


RPC_SESS_MASK = 128


class Device(ctypes.Structure):
    """TVM device strucure.

    Typically constructed using convenience function
    :meth:`tvm.runtime.device`.

    Exposes uniform interface to device-specific APIs such as CUDA or
    OpenCL.  Some properties may return None depending on whether an
    API exposes that particular property.

    NOTE!  The integer values in MASK2STR and STR2MASK *must* correspond
    to the values provided by the DLDeviceType and TVMDeviceExtType enums.
    """

    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLAOCL = 32
    kDLSDAccel = 33
    kOpenGL = 34
    kDLMicroDev = 35

    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]
    MASK2STR = {
        kDLCPU: "cpu",
        kDLCUDA: "cuda",
        kDLCUDAHost: "cuda_host",
        kDLCUDAManaged: "cuda_managed",
        kDLOpenCL: "opencl",
        kDLVulkan: "vulkan",
        kDLMetal: "metal",
        kDLVPI: "vpi",
        kDLROCM: "rocm",
        kDLROCMHost: "rocm_host",
        kDLExtDev: "ext_dev",
        kDLOneAPI: "oneapi",
        kDLWebGPU: "webgpu",
        kDLHexagon: "hexagon",
        kDLAOCL: "aocl",
        kDLSDAccel: "sdaccel",
        kOpenGL: "opengl",
        kDLMicroDev: "microdev",
    }

    STR2MASK = {
        "llvm": kDLCPU,
        "stackvm": kDLCPU,
        "cpu": kDLCPU,
        "c": kDLCPU,
        "test": kDLCPU,
        "hybrid": kDLCPU,
        "composite": kDLCPU,
        "cuda": kDLCUDA,
        "nvptx": kDLCUDA,
        "cl": kDLOpenCL,
        "opencl": kDLOpenCL,
        "sdaccel": kDLOpenCL,
        "aocl": kDLAOCL,
        "aocl_sw_emu": kDLAOCL,
        "vulkan": kDLVulkan,
        "metal": kDLMetal,
        "vpi": kDLVPI,
        "rocm": kDLROCM,
        "ext_dev": kDLExtDev,
        "hexagon": kDLHexagon,
        "webgpu": kDLWebGPU,
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

        Returns device value for cuda, rocm, and vulkan.  Returns
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

    def texture_spatial_limit(self):
        """Returns limits for textures by spatial dimensions

        Returns
        -------
        limit : int or None
            Maximum size of the texture by spatial dimensions

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
