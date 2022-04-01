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
        remote_stack_size_bytes: int = 128 * 1024,
    ):
        self._launcher = launcher
        self._session_name = session_name
        self._remote_stack_size_bytes = remote_stack_size_bytes
        self._remote_kw = remote_kw
        self._rpc = None
        self.device = None

    def __enter__(self):
        if self.device:
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
                ],
            )
            self.device = self._rpc.hexagon(0)
            return self

        except RuntimeError as exception:
            raise exception

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def upload(self, local_path: Union[str, pathlib.Path], remote_filename: str):
        """Upload a local file to the remote workspace.

        Parameters
        ----------
        local_path : str or pathlib.Path
            Path to the local file to be copied.
        remote_filename : str
            Name of the file in the remote workspace.
        """
        self._launcher.upload(local_path, remote_filename)

    def load_module(self, module: Union[str, pathlib.Path, tvm.runtime.Module]):
        """Load TVM module.

        Parameters
        ----------
        module : Union[str, pathlib.Path, tvm.runtime.Module]

            The module to load.  If `module` is a
            `tvm.runtime.Module`, it will be uploaded to the remote
            session and loaded.

            If the object passed is a string or pathlib.Path, it must
            be either a bare file name (without any path components),
            or a full path in the remote system. If it is a file name,
            the file must already have been uploaded to the remote,
            and be placed in the remote workspace.

        session : Session

            Remote session. The session must be established (via __enter__)
            prior to calling this function.

        Returns
        -------
        TVMModule :
            TVM module object.
        """
        if isinstance(module, tvm.runtime.Module):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                binary_name = "test_binary.so"
                binary_path = temp_dir / binary_name
                module.save(str(binary_path))
                self.upload(binary_path, binary_name)
                module = binary_name

        assert isinstance(module, (str, pathlib.Path)), "Invalid path type:" + str(type(module))
        return self._rpc.get_function("tvm.hexagon.load_module")(str(module))
