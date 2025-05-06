

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

_CLASS_DEVICE = None

def _set_class_device(cls):
    global _CLASS_DEVICE
    _CLASS_DEVICE = cls


def _create_device_from_tuple(cls, device_type, device_id):
    cdef DLDevice cdevice = TVMFFIDLDeviceFromIntPair(device_type, device_id)
    ret = cls.__new__(cls)
    (<Device>ret).cdevice = cdevice
    return ret


cdef class Device:
    """Device is a wrapper around DLDevice.

    Parameters
    ----------
    device_type_or_name : Union[str, int]
        The string representation of the device type

    device_id : int
        The device id
    """
    cdef DLDevice cdevice

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

    DEVICE_TYPE_TO_NAME = {
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
    }

    DEVICE_NAME_TO_TYPE = {
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
        "vulkan": kDLVulkan,
        "metal": kDLMetal,
        "vpi": kDLVPI,
        "rocm": kDLROCM,
        "ext_dev": kDLExtDev,
        "hexagon": kDLHexagon,
        "webgpu": kDLWebGPU,
    }

    def __init__(self, device_type_or_name, device_id = None):
        if isinstance(device_type_or_name, str):
            parts = device_type_or_name.split(":")
            if len(parts) < 1 or len(parts) > 2:
                raise ValueError(f"Invalid device: {device_type_or_name}")
            if parts[0] not in self.DEVICE_NAME_TO_TYPE:
                raise ValueError(f"Unknown device: {parts[0]}")
            device_type = self.DEVICE_NAME_TO_TYPE[parts[0]]
            if len(parts) == 2:
                try:
                    device_id = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid device id: {parts[1]}")
        else:
            device_type = device_type_or_name
            device_id = device_id if device_id is not None else 0
        if not isinstance(device_id, int):
            raise TypeError(f"Invalid device id: {device_id}")
        self.cdevice = TVMFFIDLDeviceFromIntPair(device_type, device_id)

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

    def __device_type_name__(self):
        return self.DEVICE_TYPE_TO_NAME[self.cdevice.device_type]

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

    @property
    def device_type(self):
        return self.cdevice.device_type

    @property
    def device_id(self):
        return self.cdevice.device_id


cdef inline object make_ret_device(TVMFFIAny result):
    ret = _CLASS_DEVICE.__new__(_CLASS_DEVICE)
    (<Device>ret).cdevice = result.v_device
    return ret


_set_class_device(Device)
