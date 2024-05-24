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
"""Detect target."""
from typing import Union

from .._ffi import get_global_func
from .._ffi.runtime_ctypes import Device
from ..runtime.ndarray import device
from . import Target


def _detect_metal(dev: Device) -> Target:
    return Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def _detect_cuda(dev: Device) -> Target:
    return Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "arch": "sm_" + dev.compute_version.replace(".", ""),
        }
    )


def _detect_rocm(dev: Device) -> Target:
    return Target(
        {
            "kind": "rocm",
            "mtriple": "amdgcn-amd-amdhsa-hcc",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def _detect_opencl(dev: Device) -> Target:
    return Target(
        {
            "kind": "opencl",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def _detect_vulkan(dev: Device) -> Target:
    f_get_target_property = get_global_func("device_api.vulkan.get_target_property")
    return Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": f_get_target_property(dev, "supports_float16"),
            "supports_int8": f_get_target_property(dev, "supports_int8"),
            "supports_int16": f_get_target_property(dev, "supports_int16"),
            "supports_int64": f_get_target_property(dev, "supports_int64"),
            "supports_8bit_buffer": f_get_target_property(dev, "supports_8bit_buffer"),
            "supports_16bit_buffer": f_get_target_property(dev, "supports_16bit_buffer"),
            "supports_storage_buffer_storage_class": f_get_target_property(
                dev, "supports_storage_buffer_storage_class"
            ),
        }
    )


def _detect_cpu(dev: Device) -> Target:  # pylint: disable=unused-argument
    """Detect the host CPU architecture."""
    return Target(
        {
            "kind": "llvm",
            "mtriple": get_global_func(
                "tvm.codegen.llvm.GetDefaultTargetTriple",
                allow_missing=False,
            )(),
            "mcpu": get_global_func(
                "tvm.codegen.llvm.GetHostCPUName",
                allow_missing=False,
            )(),
        }
    )


def detect_target_from_device(dev: Union[str, Device]) -> Target:
    """Detects Target associated with the given device. If the device does not exist,
    there will be an Error.

    Parameters
    ----------
    dev : Union[str, Device]
        The device to detect the target for.
        Supported device types: ["cuda", "metal", "rocm", "vulkan", "opencl"]

    Returns
    -------
    target : Target
        The detected target.
    """
    if isinstance(dev, str):
        dev = device(dev)
    device_type = Device.MASK2STR[dev.device_type]
    if device_type not in SUPPORT_DEVICE:
        raise ValueError(
            f"Auto detection for device `{device_type}` is not supported. "
            f"Currently only supports: {SUPPORT_DEVICE.keys()}"
        )
    if not dev.exist:
        raise ValueError(
            f"Cannot detect device `{dev}`. Please make sure the device and its driver "
            "is installed properly, and TVM is compiled with the driver"
        )
    return SUPPORT_DEVICE[device_type](dev)


SUPPORT_DEVICE = {
    "cpu": _detect_cpu,
    "cuda": _detect_cuda,
    "metal": _detect_metal,
    "vulkan": _detect_vulkan,
    "rocm": _detect_rocm,
    "opencl": _detect_opencl,
}
