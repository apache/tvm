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
"""Explicit RPC-owned devices, tensors, and streams.

A remote tensor is not a local tensor or a DLPack provider. Use ``numpy`` or
``copyto(tvm.cpu())`` to materialize its contents in the client process.
"""

import numpy as np
import tvm_ffi

import tvm.runtime

from . import _ffi_api


def _register_remote_object(type_key):
    # Keep RPC imports available for feature detection in builds without RPC.
    if tvm_ffi.get_global_func("rpc.RemoteDevice", allow_missing=True) is None:
        return lambda cls: cls
    return tvm_ffi.register_object(type_key)


@_register_remote_object("rpc.RemoteDevice")
class RemoteDevice(tvm_ffi.Object):
    """A session and a device descriptor in that session's execution endpoint."""

    @property
    def index(self):
        return self.endpoint_device.index

    @property
    def type(self):
        return self.endpoint_device.type

    def __repr__(self):
        return f"RemoteDevice({self.endpoint_device})"

    def get_attr(self, attr):
        """Query a DeviceAttrKind in the owning session."""
        return _ffi_api.DeviceGetAttr(self, attr)

    @property
    def exist(self):
        return bool(self.get_attr(0))

    @property
    def max_threads_per_block(self):
        return self.get_attr(1)

    @property
    def warp_size(self):
        return self.get_attr(2)

    @property
    def max_shared_memory_per_block(self):
        return self.get_attr(3)

    @property
    def compute_version(self):
        return self.get_attr(4)

    @property
    def device_name(self):
        return self.get_attr(5)

    @property
    def max_clock_rate(self):
        return self.get_attr(6)

    @property
    def multi_processor_count(self):
        return self.get_attr(7)

    @property
    def max_thread_dimensions(self):
        import json

        return json.loads(self.get_attr(8))

    @property
    def max_registers_per_block(self):
        return self.get_attr(9)

    @property
    def gcn_arch(self):
        return self.get_attr(10)

    @property
    def available_global_memory(self):
        return self.get_attr(15)

    def texture_spatial_limit(self):
        return self.get_attr(12)

    @property
    def api_version(self):
        return self.get_attr(11)

    @property
    def driver_version(self):
        return self.get_attr(12)

    @property
    def l2_cache_size_bytes(self):
        return self.get_attr(13)

    @property
    def total_global_memory(self):
        return self.get_attr(14)

    def empty(self, shape, dtype="float32", mem_scope=None):
        """Allocate a remote tensor through this device's session."""
        return _ffi_api.Empty(tvm_ffi.Shape(shape), tvm_ffi.dtype(dtype), self, mem_scope)

    def tensor(self, data, dtype=None):
        """Allocate a remote tensor and copy data into it synchronously."""
        if isinstance(data, RemoteTensor):
            if dtype is not None and tvm_ffi.dtype(dtype) != data.dtype:
                raise ValueError("Remote tensor copy cannot convert dtype; materialize it first")
            result = self.empty(data.shape, data.dtype)
            return result.copyfrom(data)
        local = (
            data
            if isinstance(data, tvm_ffi.Tensor) and dtype is None
            else tvm.runtime.tensor(np.asarray(data, dtype=dtype))
        )
        return self.empty(local.shape, local.dtype).copyfrom(local)

    def set_device(self):
        _ffi_api.DeviceSetDevice(self)

    def create_stream(self):
        return _ffi_api.DeviceCreateStream(self)

    def free_stream(self, stream):
        _ffi_api.DeviceFreeStream(self, stream)

    def current_stream(self):
        return _ffi_api.DeviceCurrentStream(self)

    def set_stream(self, stream=None):
        _ffi_api.DeviceSetStream(self, stream)

    def sync(self, stream=None):
        _ffi_api.DeviceSync(self, stream)


@_register_remote_object("rpc.RemoteStream")
class RemoteStream(tvm_ffi.Object):
    """A session-owned stream. Release it with the owning device's ``free_stream``."""


@_register_remote_object("rpc.RemoteTensor")
class RemoteTensor(tvm_ffi.Object):
    """Remote storage with explicit metadata and session-bound transfer operations."""

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"RemoteTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    def __dlpack__(self, *args, **kwargs):
        raise TypeError("RemoteTensor cannot export local DLPack; copyto(tvm.cpu()) first")

    def __dlpack_device__(self):
        raise TypeError("RemoteTensor has no local DLPack device; copyto(tvm.cpu()) first")

    def copyfrom(self, source):
        """Copy a local value or a tensor from the same remote session."""
        if isinstance(source, RemoteTensor):
            _ffi_api.CopyFromTo(source, self, None)
            source.device.sync()
            self.device.sync()
        else:
            if not isinstance(source, tvm_ffi.Tensor):
                source = tvm.runtime.tensor(np.asarray(source, dtype=str(self.dtype)))
            if tuple(source.shape) != tuple(self.shape):
                raise ValueError("Tensor copy shape mismatch")
            if str(source.dtype) != str(self.dtype):
                raise ValueError("Tensor copy dtype mismatch")
            _ffi_api.CopyFromLocal(source, self)
        return self

    def copyto(self, target):
        """Synchronously copy into storage or allocate on an explicit device."""
        if isinstance(target, RemoteDevice):
            target = target.empty(self.shape, self.dtype)
        if isinstance(target, RemoteTensor):
            return target.copyfrom(self)
        if isinstance(target, tvm.runtime.Device):
            is_cpu = target.dlpack_device_type() == tvm.runtime.cpu().dlpack_device_type()
            local = tvm.runtime.empty(
                self.shape, self.dtype, target if is_cpu else tvm.runtime.cpu()
            )
            _ffi_api.CopyToLocal(self, local)
            if is_cpu:
                return local
            return local.copyto(target)
        _ffi_api.CopyToLocal(self, target)
        return target

    def numpy(self):
        """Materialize in client memory and convert to NumPy."""
        return self.copyto(tvm.runtime.cpu()).numpy()

    def _create_view(self, shape, dtype=None, relative_byte_offset=0):
        return _ffi_api.CreateView(
            self, tvm_ffi.Shape(shape), tvm_ffi.dtype(dtype or self.dtype), relative_byte_offset
        )

    def __setitem__(self, key, value):
        if (
            not isinstance(key, slice)
            or key.start is not None
            or key.stop is not None
            or key.step not in (None, 1)
        ):
            raise ValueError("RemoteTensor only supports full-slice assignment")
        self.copyfrom(value)


def time_evaluator(
    module,
    func_name,
    device,
    number=10,
    repeat=1,
    min_repeat_ms=0,
    limit_zero_time_iterations=100,
    cooldown_interval_ms=0,
    repeats_to_cooldown=1,
    cache_flush_bytes=0,
    f_preproc="",
):
    """Create an evaluator after validating the module and remote device session."""
    return module.time_evaluator(
        func_name,
        device,
        number,
        repeat,
        min_repeat_ms,
        limit_zero_time_iterations,
        cooldown_interval_ms,
        repeats_to_cooldown,
        cache_flush_bytes,
        f_preproc,
    )


def virtual_machine(module, device, memory_cfg=None):
    """Initialize a remote VM with session-checked endpoint device descriptors."""
    from tvm.runtime.vm import VirtualMachine

    devices = list(device) if isinstance(device, list | tuple) else [device]
    local = [_ffi_api.ValidateDevice(module, dev) for dev in devices]
    if isinstance(memory_cfg, dict):
        memory_cfg = {
            _ffi_api.ValidateDevice(module, dev): value for dev, value in memory_cfg.items()
        }
    return VirtualMachine(module, local, memory_cfg)
