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

import logging
import os
import pickle
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from ..._ffi import get_global_func, register_func, register_object
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
        worker0_only: bool = False,
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

        worker0_only: bool
            If False (default), allocate an array on each worker.  If
            True, only allocate an array on worker0.

        Returns
        -------
        array : DRef
            The created NDArray.

        """
        if device is None:
            device = Device(device_type=0, device_id=0)
        func = self._get_cached_method("runtime.disco.empty")
        return func(ShapeTuple(shape), dtype, device, worker0_only)

    def shutdown(self):
        """Shut down the Disco session"""
        _ffi_api.SessionShutdown(self)  # type: ignore # pylint: disable=no-member

    @property
    def num_workers(self) -> int:
        """Return the number of workers in the session"""
        return _ffi_api.SessionGetNumWorkers(self)  # type: ignore # pylint: disable=no-member

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

    def import_python_module(self, module_name: str) -> None:
        """Import a python module in each worker

        This may be required before call

        Parameters
        ----------
        module_name: str

            The python module name, as it would be used in a python
            `import` statement.
        """
        if not hasattr(self, "_import_python_module"):
            self._import_python_module = self.get_global_func("runtime.disco._import_python_module")

        self._import_python_module(module_name)

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

    def copy_to_worker_0(self, host_array: NDArray, remote_array: Optional[DRef] = None) -> DRef:
        """Copy the controller-side NDArray to worker-0.

        Parameters
        ----------
        host_array : NDArray

            The array to be copied to worker-0.

        remote_array : Optiona[DRef]

            The destination NDArray on worker-0.

        Returns
        -------
        output_array: DRef

            The DRef containing the copied data on worker0, and
            NullOpt on all other workers.  If `remote_array` was
            provided, this return value is the same as `remote_array`.
            Otherwise, it is the newly allocated space.

        """
        if remote_array is None:
            remote_array = self.empty(host_array.shape, host_array.dtype, worker0_only=True)

        _ffi_api.SessionCopyToWorker0(self, host_array, remote_array)  # type: ignore # pylint: disable=no-member
        return remote_array

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
        _ffi_api.SessionInitCCL(self, ccl, ShapeTuple(device_ids))  # type: ignore # pylint: disable=no-member
        self._clear_ipc_memory_pool()

    def broadcast(self, src: Union[np.ndarray, NDArray], dst: Optional[DRef] = None) -> DRef:
        """Broadcast an array to all workers

        Parameters
        ----------
        src: Union[np.ndarray, NDArray]

            The array to be broadcasted.

        dst: Optional[DRef]

            The output array.  If None, an array matching the shape
            and dtype of `src` will be allocated on each worker.

        Returns
        -------
        output_array: DRef

            The DRef containing the broadcasted data on all workers.
            If `dst` was provided, this return value is the same as
            `dst`.  Otherwise, it is the newly allocated space.

        """
        if not isinstance(src, NDArray):
            src = _as_NDArray(src)

        if dst is None:
            dst = self.empty(src.shape, src.dtype)

        src_dref = self.copy_to_worker_0(src)
        self.broadcast_from_worker0(src_dref, dst)

        return dst

    def broadcast_from_worker0(self, src: DRef, dst: DRef) -> DRef:
        """Broadcast an array from worker-0 to all other workers.

        Parameters
        ----------
        array : DRef
            The array to be broadcasted in-place
        """
        func = self._get_cached_method("runtime.disco.broadcast_from_worker0")
        func(src, dst)

    def scatter(self, src: Union[np.ndarray, NDArray], dst: Optional[DRef] = None) -> DRef:
        """Scatter an array across all workers

        Parameters
        ----------
        src: Union[np.ndarray, NDArray]

            The array to be scattered.  The first dimension of this
            array, `src.shape[0]`, must be equal to the number of
            workers.

        dst: Optional[DRef]

            The output array.  If None, an array with compatible shape
            and the same dtype as `src` will be allocated on each
            worker.

        Returns
        -------
        output_array: DRef

            The DRef containing the scattered data on all workers.
            If `dst` was provided, this return value is the same as
            `dst`.  Otherwise, it is the newly allocated space.

        """
        assert src.shape[0] == self.num_workers

        if not isinstance(src, NDArray):
            src = _as_NDArray(src)

        if dst is None:
            dst = self.empty(src.shape[1:], src.dtype)

        src_dref = self.copy_to_worker_0(src)
        self.scatter_from_worker0(src_dref, dst)

        return dst

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

    def _clear_ipc_memory_pool(self):
        # Clear the IPC memory allocator when the allocator exists.
        name = "runtime.disco.cuda_ipc.cuda_ipc_memory_allocator_clear"
        if get_global_func(name, allow_missing=True) is not None:
            self.call_packed(self.get_global_func(name))


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

    def __init__(self, num_workers: int, entrypoint: str = "tvm.exec.disco_worker") -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.SessionProcess,  # type: ignore # pylint: disable=no-member
            num_workers,
            "runtime.disco.create_process_pool",
            entrypoint,
        )
        self._configure_structlog()

    def _configure_structlog(self) -> None:
        try:
            import structlog  # pylint: disable=import-outside-toplevel
        except ImportError:
            return

        root_logger = logging.getLogger()
        if len(root_logger.handlers) == 1 and isinstance(
            root_logger.handlers[0].formatter, structlog.stdlib.ProcessorFormatter
        ):
            stdlib_formatter = root_logger.handlers[0].formatter
        else:
            stdlib_formatter = None

        stdlib_level = root_logger.level

        full_config = (structlog.get_config(), stdlib_formatter, stdlib_level)

        config = pickle.dumps(full_config)
        func = self.get_global_func("runtime.disco._configure_structlog")
        func(config, os.getpid())


@register_func("runtime.disco._configure_structlog")
def _configure_structlog(pickled_config: bytes, parent_pid: int) -> None:
    """Configure structlog for all disco workers

    The child processes

    Parameters
    ----------
    pickled_config: bytes

        The pickled configuration for structlog

    parent_pid: int

        The PID of the main process.  This is used to restrict the
    """
    if os.getpid() == parent_pid:
        return

    import structlog  # pylint: disable=import-outside-toplevel

    full_config = pickle.loads(pickled_config)
    structlog_config, stdlib_formatter, stdlib_level = full_config

    root_logger = logging.getLogger()

    root_logger.setLevel(stdlib_level)
    if stdlib_formatter is not None:
        handler = logging.StreamHandler()
        handler.setFormatter(stdlib_formatter)
        root_logger.addHandler(handler)

    structlog.configure(**structlog_config)


@register_func("runtime.disco._import_python_module")
def _import_python_module(module_name: str) -> None:
    __import__(module_name)


REDUCE_OPS = {
    "sum": 0,
    "prod": 1,
    "min": 2,
    "max": 3,
    "avg": 4,
}
