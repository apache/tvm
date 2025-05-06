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

import tvm.ffi

from . import _ffi_api


RPC_SESS_MASK = 128


class Device(tvm.ffi.core.Device):
    """TVM device strucure."""

    def _GetDeviceAttr(self, device_type, device_id, attr_id):
        """Internal helper function to invoke runtime.GetDeviceAttr"""
        # pylint: disable=import-outside-toplevel
        return _ffi_api.GetDeviceAttr(device_type, device_id, attr_id)

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
        return self._GetDeviceAttr(self.device_type, self.device_id, 13)

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
        return self._GetDeviceAttr(self.device_type, self.device_id, 14)

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
        return self._GetDeviceAttr(self.device_type, self.device_id, 15)

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
        return _ffi_api.Device_StreamCreate(self)

    def free_raw_stream(self, stream):
        """Free a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be released.
        """
        _ffi_api.Device_StreamFree(self, stream)

    def set_raw_stream(self, stream):
        """Set a created stream handle.

        Parameters
        ----------
        stream : TVMStreamHandle
            The stream which should to be set to the device.
        """
        _ffi_api.Device_SetStream(self, stream)

    def sync(self, stream=None):
        """Synchronize until jobs finished at the context.

        Parameters
        ----------
        stream : TVMStreamHandle
            Jobs in this stream should be finished.
        """
        _ffi_api.Device_StreamSync(self, stream or 0)

    def _device_type_name_(self):
        if self.device_type >= RPC_SESS_MASK:
            tbl_id = self.device_type / RPC_SESS_MASK - 1
            dev_type = self.device_type % RPC_SESS_MASK
            return f"remote[{tbl_id}]:{Device.DEVICE_TYPE_TO_NAME[dev_type]}"
        return Device.DEVICE_TYPE_TO_NAME[self.device_type]

    def __device_type_name__(self):
        if self.device_type >= RPC_SESS_MASK:
            tbl_id = self.device_type / RPC_SESS_MASK - 1
            dev_type = self.device_type % RPC_SESS_MASK
            return f"remote[{tbl_id}]:{Device.DEVICE_TYPE_TO_NAME[dev_type]}"
        return Device.DEVICE_TYPE_TO_NAME[self.device_type]


tvm.ffi.core._set_class_device(Device)
