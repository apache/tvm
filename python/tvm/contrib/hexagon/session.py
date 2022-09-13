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

"""Defines a Session class for Hexagon devices."""

import os
import pathlib
import tempfile
from typing import Union

import tvm
from tvm import rpc as _rpc
import tvm.contrib.hexagon as hexagon
from tvm.relay.backend.executor_factory import (
    ExecutorFactoryModule,
    AOTExecutorFactoryModule,
    GraphExecutorFactoryModule,
)
from .tools import export_module


class Session:
    """Hexagon Device Session

    Parameters
    ----------
    launcher : HexagonLauncherRPC
        The launcher from which this session was started.

    remote_kw : dict
        Remote configs for RPC tracker.

    session_name : str
        Hexagon RPC session name.

    remote_stack_size_bytes : int
        The stack size of the remote device, to be passed to
        tvm.contrib.hexagon.create_hexagon_session.
    """

    def __init__(
        self,
        launcher: "HexagonLauncherRPC",
        remote_kw: dict,
        session_name: str = "hexagon-rpc",
        remote_stack_size_bytes: int = 256 * 1024,  # Min size for main thread in QuRT/sim
        rpc_receive_buffer_size_bytes: int = 256 * 1024 * 1024,  # Size for passing hexagon tests
    ):
        self._launcher = launcher
        self._session_name: str = session_name
        self._remote_stack_size_bytes: int = remote_stack_size_bytes
        self._rpc_receive_buffer_size_bytes: int = rpc_receive_buffer_size_bytes
        self._remote_kw: dict = remote_kw
        self._rpc = None
        self._requires_cpu_device = False
        self._device = None

    def __enter__(self):
        if self._rpc:
            # Already initialized
            return self

        tracker = _rpc.connect_tracker(self._remote_kw["host"], self._remote_kw["port"])
        try:
            self._rpc = tracker.request(
                self._remote_kw["key"],
                priority=self._remote_kw["priority"],
                session_timeout=self._remote_kw["timeout"],
                session_constructor_args=[
                    "tvm.contrib.hexagon.create_hexagon_session",
                    self._session_name,
                    self._remote_stack_size_bytes,
                    os.environ.get("HEXAGON_SIM_ARGS", ""),
                    self._rpc_receive_buffer_size_bytes,
                ],
            )
            return self

        except RuntimeError as exception:
            raise exception

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # close session to the tracker
        del self._rpc

    @property
    def device(self):
        """Session device."""

        if self._device is not None:
            return self._device

        if self._requires_cpu_device:
            self._device = self._rpc.cpu(0)
        else:
            self._device = self._rpc.hexagon(0)

        return self._device

    def get_function(self, name):
        return self._rpc.get_function(name)

    def upload(self, local_path: Union[str, pathlib.Path], remote_filename: str) -> pathlib.Path:
        """Upload a local file to the remote workspace.

        Parameters
        ----------
        local_path : str or pathlib.Path
            Path to the local file to be copied.
        remote_filename : str
            Name of the file in the remote workspace.

        Returns
        -------
        pathlib.Path :
            Uploaded file remote path.
        """
        return self._launcher.upload(local_path, remote_filename)

    def load_module(self, module: Union[str, pathlib.Path, tvm.runtime.Module]):
        """Load TVM module.

        The session must be established (via __enter__) prior to
        calling this function.

        Parameters
        ----------
        module : Union[str, pathlib.Path, tvm.runtime.Module]

            The module to load.  If `module` is a
            `tvm.runtime.Module`, it will be uploaded to the remote
            session and loaded.

            If the object passed is a string or pathlib.Path, it must
            be a full path in the remote system.

        Returns
        -------
        TVMModule :
            TVM module object.
        """

        assert self._rpc is not None, "Hexagon session must be started using __enter__ prior to use"

        if isinstance(module, tvm.runtime.Module):
            with tempfile.TemporaryDirectory() as temp_dir:
                binary_name = "test_binary.so"
                binary_path = export_module(module, temp_dir, binary_name)
                remote_file_path = self.upload(binary_path, binary_name)
        else:
            remote_file_path = module

        assert isinstance(remote_file_path, (str, pathlib.Path)), "Invalid path type:" + str(
            type(remote_file_path)
        )
        return self._rpc.get_function("tvm.hexagon.load_module")(str(remote_file_path))

    def get_graph_executor(
        self,
        graph_json: str,
        module_name: Union[str, pathlib.Path, tvm.runtime.Module],
    ):
        """Create a local GraphModule which consumes a remote libmod.

        The session must be established (via __enter__) prior to
        calling this function.

        Parameters
        ----------
        module_name : Union[str, pathlib.Path, tvm.runtime.Module]
            The remote module filename, following the same restrictions
            as `load_module`.
        graph_json : str
            The string with the graph JSON.

        Returns
        -------
        GraphModule :
            Runtime graph module that can be used to execute the graph.

        """

        graph_mod = self.load_module(module_name)
        self._set_device_type(graph_mod)
        return tvm.contrib.graph_executor.create(graph_json, graph_mod, self.device)

    def get_aot_executor(
        self,
        module_file: Union[str, pathlib.Path],
    ):
        """Create a local GraphModule which consumes a remote libmod.
        The session must be established (via __enter__) prior to
        calling this function.
        Parameters
        ----------
        module_file : Union[str, pathlib.Path]
            The remote module filename, following the same restrictions
            as `load_module`. The filename should be an absolute path.
        Returns
        -------
        GraphModule :
            Runtime graph module that can be used to execute the graph.
        """
        aot_mod = self.load_module(module_file)
        return tvm.runtime.executor.AotModule(aot_mod["default"](self.device))

    def get_graph_debug_executor(
        self,
        graph_json: str,
        module_name: Union[str, pathlib.Path, tvm.runtime.Module],
        dump_root: Union[str, pathlib.Path] = None,
    ):
        """Create a local GraphModuleDebug which consumes a remote libmod.

        Parameters
        ----------
        graph_json : str
            The string with the graph JSON.
         module_name : Union[str, pathlib.Path, tvm.runtime.Module]
            The remote module filename, following the same restrictions
            as `load_module`.
        session : Session
            Remote session. The session must be established (via __enter__)
            prior to calling this function.

        Returns
        -------
        GraphModuleDebug :
            Runtime debug graph module that can be used to debug the graph.
        """

        graph_debug_mod = self.load_module(module_name)
        self._set_device_type(graph_debug_mod)
        return tvm.contrib.debugger.debug_executor.create(
            graph_json, graph_debug_mod, self.device, dump_root=str(dump_root)
        )

    def get_executor_from_factory(self, module: ExecutorFactoryModule):
        """Create a local GraphModule which consumes a remote libmod.

        Parameters
        ----------

        module : ExecutorFactoryModule

            The module to upload to the remote
            session and load.
        """
        if isinstance(module, AOTExecutorFactoryModule):
            return self._aot_executor_from_factory(module)
        if isinstance(module, GraphExecutorFactoryModule):
            return self._graph_executor_from_factory(module)

        raise TypeError(f"Unsupported executor type: {type(module)}")

    def _set_device_type(self, module: Union[str, pathlib.Path, GraphExecutorFactoryModule]):
        """Set session device type(hexagon, cpu) based on target in module.

        Parameters
        ----------

        module: TVMModule
            TVM module object.
        """
        # for cases when module is a single schedule without target attribute.
        if not hasattr(module, "target"):
            self._requires_cpu_device = False
        else:
            assert len(module.target) == 1
            for target in module.target:
                target_type = str(target).split()[0]

            if target_type == "llvm":
                self._requires_cpu_device = True
            else:
                self._requires_cpu_device = False

    def _graph_executor_from_factory(
        self,
        module: Union[str, pathlib.Path, GraphExecutorFactoryModule],
    ):
        """Create a local GraphModule which consumes a remote libmod.

        The session must be established (via __enter__) prior to
        calling this function.

        Parameters
        ----------

        module : GraphExecutorFactoryModule

            The graph executor module to upload to the remote and load.
            This will typically be the output of `tvm.relay.build`,
            when passing `executor=Executor("graph")`.

        Returns
        -------
        GraphModule :
            Runtime graph module that can be used to execute the graph.

        """
        return self.get_graph_executor(module.get_graph_json(), module.get_lib())

    def _aot_executor_from_factory(
        self,
        module: Union[str, pathlib.Path, AOTExecutorFactoryModule],
    ):
        """Create a local GraphModule which consumes a remote libmod.

        The session must be established (via __enter__) prior to
        calling this function.

        Parameters
        ----------

        module : AOTExecutorFactoryModule

            The graph executor module to upload to the remote and load.
            This will typically be the output of `tvm.relay.build`,
            when passing `executor=Executor("aot")`.

        Returns
        -------
        GraphModule :
            Runtime graph module that can be used to execute the graph.

        """

        hexagon_arch = set(
            target.mcpu.replace("hexagon", "")
            for target in module.target
            if "hexagon" in target.keys
        )

        self._set_device_type(module)

        for target in module.target:
            target_type = str(target).split()[0]

        assert hexagon_arch, "No hexagon target architecture found"
        assert len(hexagon_arch) == 1, f"Inconsistent hexagon architecture found, {hexagon_arch}"
        hexagon_arch = hexagon_arch.pop()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            binary_name = "test_binary.so"
            binary_path = temp_dir / binary_name

            if target_type == "hexagon":
                module.export_library(
                    str(binary_path),
                    fcompile=hexagon.create_aot_shared,
                    hexagon_arch=hexagon_arch,
                )
            elif target_type == "llvm":
                module.export_library(
                    str(binary_path),
                    cc=hexagon.hexagon_clang_plus(),
                )
            else:
                raise ValueError(
                    f"Incorrect Target kind.\n"
                    f"Target kind should be from these options: [hexagon, llvm]."
                )

            remote_file_path = self.upload(binary_path, binary_name)

        return self.get_aot_executor(remote_file_path)
