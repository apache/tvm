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
import json

import tvm_ffi

from . import _ffi_api


class Device(tvm_ffi.core.Device):
    """TVM device strucure."""

    def _GetDeviceAttr(self, device_type, device_id, name):
        """Internal helper function to invoke runtime.GetDeviceAttr"""
        # pylint: disable=import-outside-toplevel
        return _ffi_api.GetDeviceAttr(device_type, device_id, name)

    def get_attr(self, name):
        """Query a named device attribute, returning None when unsupported.

        ``exist`` returns a bool.  ``max_thread_dimensions`` returns a JSON
        string; the property of the same name decodes it to a list.
        Vulkan exposes numeric version values as ``vulkan_api_version`` and
        ``vulkan_driver_version``, separately from its formatted version strings.
        """
        return self._GetDeviceAttr(self.dlpack_device_type(), self.index, name)

    @property
    def exist(self):
        """Whether this device exists.

        Returns True if TVM has support for the device, if the
        physical device is present, and the device is accessible
        through appropriate drivers (e.g. CUDA/Vulkan).

        Returns
        -------
        exist : bool
            True if the device exists

        """
        return self.get_attr("exist")

    @property
    def max_threads_per_block(self):
        """Maximum number of threads on each block.

        Returns device value for CUDA, Metal, ROCm, OpenCL, and Vulkan
        devices.  Returns None for all other devices.

        Returns
        -------
        max_threads_per_block : int or None
            The number of threads on each block

        """
        return self.get_attr("max_threads_per_block")

    @property
    def warp_size(self):
        """Number of threads that execute concurrently.

        Returns device value for CUDA, ROCm, and Vulkan.  Returns
        1 for Metal and OpenCL devices, regardless of the physical
        device.  Returns
        None for all other devices.

        Returns
        -------
        warp_size : int or None
            Number of threads that execute concurrently

        """
        return self.get_attr("warp_size")

    @property
    def max_shared_memory_per_block(self):
        """Total amount of shared memory per block in bytes.

        Returns device value for CUDA, ROCm, OpenCL, and Vulkan.
        Returns None for
        all other devices.

        Returns
        -------
        max_shared_memory_per_block : int or None
            Total amount of shared memory per block in bytes

        """
        return self.get_attr("max_shared_memory_per_block")

    @property
    def compute_version(self):
        """Get compute version number as string.

        Returns maximum API version (e.g. CUDA/OpenCL/Vulkan)
        supported by the device.

        Returns device value for CUDA, ROCm, OpenCL, and
        Vulkan. Returns
        None for all other devices.

        Returns
        -------
        version : str or None
            The version string in `major.minor` format.

        """
        return self.get_attr("compute_version")

    @property
    def device_name(self):
        """Return the vendor-specific name of device.

        Returns device value for CUDA, ROCm, OpenCL, and Vulkan.
        Returns None for
        all other devices.

        Returns
        -------
        device_name : str or None
            The name of the device.

        """
        return self.get_attr("device_name")

    @property
    def max_clock_rate(self):
        """Return the max clock frequency of device (kHz).

        Returns device value for CUDA, ROCm, and OpenCL.  Returns
        None for all
        other devices.

        Returns
        -------
        max_clock_rate : int or None
            The maximum clock frequency of the device (kHz)

        """
        return self.get_attr("max_clock_rate")

    @property
    def multi_processor_count(self):
        """Return the number of compute units in the device.

        Returns device value for CUDA, ROCm, and OpenCL.  Returns
        None for all
        other devices.

        Returns
        -------
        multi_processor_count : int or None
            Thee number of compute units in the device

        """
        return self.get_attr("multi_processor_count")

    @property
    def max_thread_dimensions(self):
        """Return the maximum size of each thread axis

        Returns device value for CUDA, ROCm, OpenCL, and Vulkan.
        Returns None for
        all other devices.

        Returns
        -------
        dims: List of int, or None
            The maximum length of threadIdx.x, threadIdx.y, threadIdx.z

        """
        value = self.get_attr("max_thread_dimensions")
        return json.loads(value) if value is not None else None

    @property
    def api_version(self):
        """Returns version number of the SDK used to compile TVM.

        For example, CUDA_VERSION for CUDA or VK_HEADER_VERSION for
        Vulkan.

        Returns device value for CUDA, ROCm, OpenCL, and Vulkan.
        Returns None for
        all other devices.

        Returns
        -------
        version : int or None
            The version of the SDK

        """
        return self.get_attr("api_version")

    @property
    def driver_version(self):
        """Returns version number of the driver

        Returns driver vendor's internal version number.
        (e.g. "450.408.256" for nvidia-driver-450)

        Returns device value for opencl and vulkan.  Returns None for all other
        devices.

        Returns
        -------
        version : str or None
            The version string in `major.minor.patch` format.

        """
        return self.get_attr("driver_version")

    @property
    def l2_cache_size_bytes(self):
        """Return the size of the device L2 cache in bytes

        Supported devices include CUDA/ROCM/OpenCL.

        Returns
        -------
        l2_cache_size_bytes : int or None
            The size of the device L2 cache in bytes returned by device runtime API.
            Return None if the device does not support this feature.

        Note
        ----
        The value returned by opencl's API is smaller than actual device L2 cache size.
        """
        return self.get_attr("l2_cache_size_bytes")

    @property
    def total_global_memory(self):
        """Return size of the total global memory.

        Supported devices include CUDA/ROCm/Metal/OpenCL.

        Returns
        -------
        total_global_memory : int or None
            Return the total size of global memory on device in bytes.
            Return None if the device does not support this feature.
        """
        return self.get_attr("total_global_memory")

    @property
    def max_registers_per_block(self):
        """Return the block register limit, or None when unsupported."""
        return self.get_attr("max_registers_per_block")

    @property
    def gcn_arch(self):
        """Return the ROCm architecture name, or None when unsupported."""
        return self.get_attr("gcn_arch")

    @property
    def available_global_memory(self):
        """Return size of the available global memory.

        Supported devices include CUDA.

        Returns
        -------
        available_global_memory : int or None
            Return the amount of unallocated global memory on device in bytes.
            Return None if the device does not support this feature.
        """
        return self.get_attr("available_global_memory")

    def create_raw_stream(self):
        """Create a new runtime stream at the context.

        User should free the stream after use.

        Returns
        -------
        stream : TVMStreamHandle
            The created runtime stream.
        """
        return _ffi_api.DeviceStreamCreate(self)

    def free_raw_stream(self, stream):
        """Free a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be released.
        """
        _ffi_api.DeviceStreamFree(self, stream)

    def set_raw_stream(self, stream):
        """Set a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be set to the device.
        """
        _ffi_api.DeviceSetStream(self, stream)

    def sync(self, stream=None):
        """Synchronize until jobs finished at the context.

        Parameters
        ----------
        stream : TVMStreamHandle
            Jobs in this stream should be finished.
        """
        _ffi_api.DeviceStreamSync(self, stream or 0)

    def __device_type_name__(self):
        return Device._DEVICE_TYPE_TO_NAME[self.dlpack_device_type()]


tvm_ffi.core._set_class_device(Device)
