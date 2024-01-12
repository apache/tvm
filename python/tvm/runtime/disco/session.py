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
"""This module defines a Session in Disco. Session is the primary interface that users interact
with the distributed runtime.
"""
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from ..._ffi import register_object
from ..._ffi.runtime_ctypes import Device
from ..container import ShapeTuple
from ..ndarray import NDArray
from ..ndarray import array as _as_NDArray
from ..object import Object
from . import _ffi_api, process_pool  # pylint: disable=unused-import


@register_object("runtime.disco.DRef")
class DRef(Object):
    """An object that exists on all workers. The controller process assigns a unique "register id"
    to each object, and the worker process uses this id to refer to the object residing on itself.
    """

    def debug_get_from_remote(self, worker_id: int) -> Any:
        """Get the value of a DRef from a remote worker. It is only used for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be fetched from.

        Returns
        -------
        value : object
            The value of the register.
        """
        return _ffi_api.DRefDebugGetFromRemote(self, worker_id)  # type: ignore # pylint: disable=no-member

    def debug_copy_from(
        self,
        worker_id: int,
        value: Union[np.ndarray, NDArray],
    ) -> None:
        """Copy an NDArray value to remote for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be copied to.
        value : Union[numpy.ndarray, NDArray]
            The value to be copied.
        """
        if not isinstance(value, NDArray):
            value = _as_NDArray(value)
        return _ffi_api.DRefDebugCopyFrom(self, worker_id, value)  # type: ignore # pylint: disable=no-member


class DPackedFunc(DRef):
    """A PackedFunc in a Disco session."""

    def __init__(self, dref: DRef, session: "Session") -> None:
        self.handle = dref.handle
        dref.handle = None
        self.session = session

    def __call__(self, *args) -> DRef:
        return self.session.call_packed(self, *args)


class DModule(DRef):
    """A Module in a Disco session."""

    def __init__(self, dref: DRef, session: "Session") -> None:
        self.handle = dref.handle
        del dref.handle
        self.session = session

    def __getitem__(self, name: str) -> DPackedFunc:
        func = self.session._get_cached_method("runtime.ModuleGetFunction")
        return DPackedFunc(func(self, name, False), self.session)


@register_object("runtime.disco.Session")
class Session(Object):
    """A Disco interactive session. It allows users to interact with the Disco command queue with
    various PackedFunc calling convention."""

    def _get_cached_method(self, name: str) -> Callable:
        if "_cache" not in self.__dict__:
            cache = self._cache = {}  # pylint: disable=attribute-defined-outside-init
        else:
            cache = self._cache
        if name not in cache:
            func = cache[name] = self.get_global_func(name)
        else:
            func = cache[name]
        return func

    def empty(
        self,
        shape: Sequence[int],
        dtype: str,
        device: Optional[Device] = None,
    ) -> DRef:
        """Create an empty NDArray on all workers and attach them to a DRef.

        Parameters
        ----------
        shape : tuple of int
            The shape of the NDArray.
        dtype : str
            The data type of the NDArray.
        device : Optional[Device] = None
            The device of the NDArray.

        Returns
        -------
        array : DRef
            The created NDArray.
        """
        if device is None:
            device = Device(device_type=0, device_id=0)
        func = self._get_cached_method("runtime.disco.empty")
        return func(ShapeTuple(shape), dtype, device)

    def get_global_func(self, name: str) -> DRef:
        """Get a global function on workers.

        Parameters
        ----------
        name : str
            The name of the global function.

        Returns
        -------
        func : DRef
            The global packed function
        """
        return DPackedFunc(_ffi_api.SessionGetGlobalFunc(self, name), self)  # type: ignore # pylint: disable=no-member

    def call_packed(self, func: DRef, *args) -> DRef:
        """Call a PackedFunc on workers providing variadic arguments.

        Parameters
        ----------
        func : PackedFunc
            The function to be called.
        *args : various types
            In the variadic arguments, the supported types include:
            - integers and floating point numbers;
            - DLDataType;
            - DLDevice;
            - str (std::string in C++);
            - DRef.

        Returns
        -------
        return_value : various types
            The return value of the function call.

        Notes
        -----
        Examples of unsupported types:
        - NDArray, DLTensor,;
        - TVM Objects, including PackedFunc, Module and String.
        """
        return _ffi_api.SessionCallPacked(self, 0, 0, func, *args)  # type: ignore # pylint: disable=no-member

    def _sync_worker(self, worker_id: int) -> None:
        """Synchronize the controller with a worker, and it will wait until the worker finishes
        executing all the existing instructions. This function is usually used for worker-0, because
        it is the only worker that is assumed to collocate with the controller. Syncing with other
        workers may not be supported and should only be used for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be synced with.
        """
        return _ffi_api.SessionSyncWorker(self, worker_id)  # type: ignore # pylint: disable=no-member

    def sync_worker_0(self) -> None:
        """Synchronize the controller with worker-0, and it will wait until the worker-0 finishes
        executing all the existing instructions."""
        return self._sync_worker(0)

    def copy_from_worker_0(self, host_array: NDArray, remote_array: DRef) -> None:
        """Copy an NDArray from worker-0 to the controller-side NDArray.

        Parameters
        ----------
        host_array : numpy.ndarray
            The array to be copied to worker-0.
        remote_array : NDArray
            The NDArray on worker-0.
        """
        return _ffi_api.SessionCopyFromWorker0(self, host_array, remote_array)  # type: ignore # pylint: disable=no-member

    def copy_to_worker_0(self, host_array: NDArray, remote_array: DRef) -> None:
        """Copy the controller-side NDArray to worker-0.

        Parameters
        ----------
        host_array : numpy.ndarray
            The array to be copied from worker-0.
        remote_array : NDArray
            The NDArray on worker-0.
        """
        return _ffi_api.SessionCopyToWorker0(self, host_array, remote_array)  # type: ignore # pylint: disable=no-member

    def load_vm_module(
        self,
        path: str,
        device: Optional[Device] = None,
    ) -> DModule:
        """Load a VM module from a file.

        Parameters
        ----------
        path : str
            The path to the VM module file.
        device : Optional[Device] = None
            The device to load the VM module to. Default to the default device of each worker.

        Returns
        -------
        module : DModule
            The loaded VM module.
        """
        if device is None:
            device = Device(device_type=0, device_id=0)
        func = self._get_cached_method("runtime.disco.load_vm_module")
        return DModule(func(path, device), self)

    def init_ccl(self, ccl: str, *device_ids):
        """Initialize the underlying communication collective library.

        Parameters
        ----------
        ccl : str
            The name of the communication collective library. Currently supported libraries are:
            - nccl
            - rccl
            - mpi
        *device_ids : int
            The device IDs to be used by the underlying communication library.
        """
        assert ccl in ("nccl", "rccl"), f"Unsupported CCL backend: {ccl}"
        return _ffi_api.SessionInitCCL(self, ccl, ShapeTuple(device_ids))  # type: ignore # pylint: disable=no-member

    def broadcast_from_worker0(self, src: DRef, dst: DRef) -> DRef:
        """Broadcast an array from worker-0 to all other workers.

        Parameters
        ----------
        array : DRef
            The array to be broadcasted in-place
        """
        func = self._get_cached_method("runtime.disco.broadcast_from_worker0")
        func(src, dst)

    def scatter_from_worker0(self, from_array: DRef, to_array: DRef) -> None:
        """Scatter an array from worker-0 to all other workers.

        Parameters
        ----------
        from_array : DRef
            The array to be scattered from.
        to_array : DRef
            The array to be scattered to.
        """
        func = self._get_cached_method("runtime.disco.scatter_from_worker0")
        func(from_array, to_array)

    def gather_to_worker0(self, from_array: DRef, to_array: DRef) -> None:
        """Gather an array from all other workers to worker-0.

        Parameters
        ----------
        from_array : DRef
            The array to be gathered from.
        to_array : DRef
            The array to be gathered to.
        """
        func = self._get_cached_method("runtime.disco.gather_to_worker0")
        func(from_array, to_array)

    def allreduce(
        self,
        src: DRef,
        dst: DRef,
        op: str = "sum",  # pylint: disable=invalid-name
    ) -> DRef:
        """Perform an allreduce operation on an array.

        Parameters
        ----------
        array : DRef
            The array to be reduced.
        op : str = "sum"
            The reduce operation to be performed. Available options are:
            - "sum"
            - "prod"
            - "min"
            - "max"
            - "avg"
        """
        if op not in REDUCE_OPS:
            raise ValueError(f"Unsupported reduce op: {op}. Available ops are: {REDUCE_OPS.keys()}")
        op = ShapeTuple([REDUCE_OPS[op]])
        func = self._get_cached_method("runtime.disco.allreduce")
        func(src, op, dst)

    def allgather(
        self,
        src: DRef,
        dst: DRef,
    ) -> DRef:
        """Perform an allgather operation on an array.

        Parameters
        ----------
        src : DRef
            The array to be gathered from.
        dst : DRef
            The array to be gathered to.
        """
        func = self._get_cached_method("runtime.disco.allgather")
        func(src, dst)


@register_object("runtime.disco.ThreadedSession")
class ThreadedSession(Session):
    """A Disco session backed by multi-threading."""

    def __init__(self, num_workers: int) -> None:
        """Create a disco session backed by multiple threads in the same process."""
        self.__init_handle_by_constructor__(
            _ffi_api.SessionThreaded,  # type: ignore # pylint: disable=no-member
            num_workers,
        )


@register_object("runtime.disco.ProcessSession")
class ProcessSession(Session):
    """A Disco session backed by pipe-based multi-processing."""

    def __init__(self, num_workers: int, entrypoint: str) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.SessionProcess,  # type: ignore # pylint: disable=no-member
            num_workers,
            "runtime.disco.create_process_pool",
            entrypoint,
        )


REDUCE_OPS = {
    "sum": 0,
    "prod": 1,
    "min": 2,
    "max": 3,
    "avg": 4,
}
