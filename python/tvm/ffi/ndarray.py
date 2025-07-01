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
"""NDArray related objects and functions."""

from numbers import Integral
from . import core
from .core import Device, NDArray, from_dlpack
from . import registry
from . import _ffi_api


@registry.register_object("ffi.Shape")
class Shape(tuple, core.PyNativeObject):
    """Shape object that is possibly returned by FFI call."""

    def __new__(cls, content):
        if any(not isinstance(x, Integral) for x in content):
            raise ValueError("Shape must be a tuple of integers")
        val = tuple.__new__(cls, content)
        val.__init_tvm_ffi_object_by_constructor__(_ffi_api.Shape, *content)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj):
        """Construct from a given tvm object."""
        content = core._shape_obj_get_py_tuple(obj)
        val = tuple.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val


def device(dev_type, dev_id=0):
    """Construct a TVM  FFIdevice with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev: tvm.ffi.Device

    Examples
    --------
    Device can be used to create reflection of device by
    string representation of the device type.

    .. code-block:: python

      assert tvm.ffi.device("cuda:0") == tvm.ffi.cuda(1)
      assert tvm.ffi.device("cpu", 0) == tvm.ffi.cpu(0)
    """
    if isinstance(dev_type, str):
        dev_type = dev_type.split(" ")[0]
    return core._CLASS_DEVICE(dev_type, dev_id)


def cpu(dev_id=0):
    """Construct a CPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLCPU, dev_id)


def cuda(dev_id=0):
    """Construct a CUDA GPU device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLCUDA, dev_id)


def rocm(dev_id=0):
    """Construct a ROCM device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLROCM, dev_id)


def opencl(dev_id=0):
    """Construct a OpenCL device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLOpenCL, dev_id)


def metal(dev_id=0):
    """Construct a metal device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLMetal, dev_id)


def vpi(dev_id=0):
    """Construct a VPI simulated device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLVPI, dev_id)


def vulkan(dev_id=0):
    """Construct a Vulkan device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLVulkan, dev_id)


def ext_dev(dev_id=0):
    """Construct a extension device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device

    Note
    ----
    This API is reserved for quick testing of new
    device by plugin device API as ext_dev.
    """
    return device(Device.kDLExtDev, dev_id)


def hexagon(dev_id=0):
    """Construct a Hexagon device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLHexagon, dev_id)


def webgpu(dev_id=0):
    """Construct a webgpu device.

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    dev : Device
        The created device
    """
    return device(Device.kDLWebGPU, dev_id)


__all__ = [
    "from_dlpack",
    "NDArray",
    "device",
    "cpu",
    "cuda",
    "rocm",
    "opencl",
    "metal",
    "vpi",
    "vulkan",
    "ext_dev",
    "hexagon",
    "webgpu",
]
