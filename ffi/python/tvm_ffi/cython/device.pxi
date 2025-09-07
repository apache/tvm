

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
from enum import IntEnum

_CLASS_DEVICE = None

def _set_class_device(cls):
    global _CLASS_DEVICE
    _CLASS_DEVICE = cls


def _create_device_from_tuple(cls, device_type, device_id):
    cdef DLDevice cdevice = TVMFFIDLDeviceFromIntPair(device_type, device_id)
    ret = cls.__new__(cls)
    (<Device>ret).cdevice = cdevice
    return ret


class DLDeviceType(IntEnum):
    """The enum that maps to DLDeviceType."""
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


cdef class Device:
    """Device represents a device in the ffi system.

    Device is a thin wrapper around DLDevice in DLPack standard.

    Parameters
    ----------
    device_type : Union[str, int]
        The string representation of the device type

    index : int
        The device id

    Examples
    --------
    You can use `tvm_ffi.device` function to create a `Device`.

    .. code-block:: python

      assert tvm_ffi.device("cuda:0") == tvm_ffi.device("cuda", 0)
      assert tvm_ffi.device("cpu:0") == tvm_ffi.device("cpu", 0)
    """
    cdef DLDevice cdevice

    _DEVICE_TYPE_TO_NAME = {
      DLDeviceType.kDLCPU: "cpu",
      DLDeviceType.kDLCUDA: "cuda",
      DLDeviceType.kDLCUDAHost: "cuda_host",
      DLDeviceType.kDLCUDAManaged: "cuda_managed",
      DLDeviceType.kDLOpenCL: "opencl",
      DLDeviceType.kDLVulkan: "vulkan",
      DLDeviceType.kDLMetal: "metal",
      DLDeviceType.kDLVPI: "vpi",
      DLDeviceType.kDLROCM: "rocm",
      DLDeviceType.kDLROCMHost: "rocm_host",
      DLDeviceType.kDLExtDev: "ext_dev",
      DLDeviceType.kDLOneAPI: "oneapi",
      DLDeviceType.kDLWebGPU: "webgpu",
      DLDeviceType.kDLHexagon: "hexagon",
    }

    _DEVICE_NAME_TO_TYPE = {
        "llvm": DLDeviceType.kDLCPU,
        "cpu": DLDeviceType.kDLCPU,
        "c": DLDeviceType.kDLCPU,
        "test": DLDeviceType.kDLCPU,
        "cuda": DLDeviceType.kDLCUDA,
        "nvptx": DLDeviceType.kDLCUDA,
        "cl": DLDeviceType.kDLOpenCL,
        "opencl": DLDeviceType.kDLOpenCL,
        "vulkan": DLDeviceType.kDLVulkan,
        "metal": DLDeviceType.kDLMetal,
        "vpi": DLDeviceType.kDLVPI,
        "rocm": DLDeviceType.kDLROCM,
        "ext_dev": DLDeviceType.kDLExtDev,
        "hexagon": DLDeviceType.kDLHexagon,
        "webgpu": DLDeviceType.kDLWebGPU,
    }

    def __init__(self, device_type, index = None):
        device_type_or_name = device_type
        index = index if index is not None else 0
        if isinstance(device_type_or_name, str):
            # skip suffix annotations
            device_type_or_name = device_type_or_name.split(" ")[0]
            parts = device_type_or_name.split(":")
            if len(parts) < 1 or len(parts) > 2:
                raise ValueError(f"Invalid device: {device_type_or_name}")
            if parts[0] not in self._DEVICE_NAME_TO_TYPE:
                raise ValueError(f"Unknown device: {parts[0]}")
            device_type = self._DEVICE_NAME_TO_TYPE[parts[0]]
            if len(parts) == 2:
                try:
                    index = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid device index: {parts[1]}")
        else:
            device_type = device_type_or_name
        if not isinstance(index, int):
            raise TypeError(f"Invalid device index: {index}")
        self.cdevice = TVMFFIDLDeviceFromIntPair(device_type, index)

    def __reduce__(self):
        cls = type(self)
        return (_create_device_from_tuple, (cls, self.cdevice.device_type, self.cdevice.device_id))

    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return (
            self.cdevice.device_type == (<Device>other).cdevice.device_type
            and self.cdevice.device_id == (<Device>other).cdevice.device_id
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        cdef int dev_type = self.cdevice.device_type
        name = self.__device_type_name__()
        index = self.cdevice.device_id
        return f"{name}:{index}"

    def __repr__(self):
        cdef int dev_type = self.cdevice.device_type
        name = self.__device_type_name__()
        index = self.cdevice.device_id
        return f"device(type='{name}', index={index})"

    def __hash__(self):
        return hash((self.cdevice.device_type, self.cdevice.device_id))


    def __device_type_name__(self):
        return self._DEVICE_TYPE_TO_NAME[self.cdevice.device_type]

    @property
    def type(self):
        """String representation of the device type."""
        return self.__device_type_name__()

    @property
    def index(self):
        """The device index."""
        return self.cdevice.device_id

    def dlpack_device_type(self):
        """The device type int code used in the DLPack specification.
        """
        return self.cdevice.device_type


cdef inline object make_ret_device(TVMFFIAny result):
    ret = _CLASS_DEVICE.__new__(_CLASS_DEVICE)
    (<Device>ret).cdevice = result.v_device
    return ret


_set_class_device(Device)
