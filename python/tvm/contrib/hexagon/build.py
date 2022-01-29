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

"""Defines top-level glue functions for building Hexagon."""

import pathlib
import os
import subprocess
from typing import Union
import stat
import datetime

import tvm
from ..._ffi import libinfo
from .session import Session


RPC_SERVER_FILES = ["tvm_rpc_android", "libtvm_runtime.so", "android_bash.sh"]

HEXAGON_FILES = ["libhexagon_rpc_skel.so"]

HEXAGON_RPC_DIR = None

ANDROID_HEXAGON_TEST_BASE_DIR = pathlib.Path("/data/local/tmp/hexagon_test")


def get_hexagon_rpc_dir() -> pathlib.Path:
    """Find the Hexagon library.

    Returns
    -------
    str :
        The path to the Hexagon library
    """
    global HEXAGON_RPC_DIR
    if HEXAGON_RPC_DIR is None:
        for path in libinfo.find_lib_path():
            rpc_dir = os.path.join(os.path.dirname(path), "hexagon_api_output")
            if os.path.isdir(rpc_dir):
                HEXAGON_RPC_DIR = rpc_dir
                break
        else:
            raise "hexagon_rpc was not found."
    return pathlib.Path(HEXAGON_RPC_DIR)


class HexagonLauncher:
    """Hexagon Launcher"""

    def __init__(self, serial_number: str, workspace_size_gb: int = 1):
        """Configure a new HexagonLauncher

        Parameters
        ----------
        serial_number : str
            Android device serial number from android 'adb' command.
        """
        # Hexagon RPCSession
        self.session = None

        self._serial_number = serial_number
        self._adb_device_sub_cmd = ["adb", "-s", self._serial_number]
        self._mod = None
        self._workspace = None
        self._workspace_max_size_mb = workspace_size_gb * 1024

    HEXAGON_REMOTE_DEVICE_KEY = "hexagon-dev"

    def android_run_rpc(
        self,
        workspace_dir: Union[str, pathlib.Path] = None,
        rpc_server_port: int = 7070,
        rpc_tracker_host: str = "0.0.0.0",
        rpc_tracker_port: int = 9190,
    ):
        """Upload Android artifacts and run RPC server on Android.

        Parameters
        ----------
        workspace_dir : Union[str, pathlib.Path]
            Workspace directory used on Android to upload artifacts.

        rpc_server_port : int
            Android RPC server port number

        rpc_tracker_host : str
            RPC tracker IP on host

        rpc_tracker_port : int
            RPC tracker port on host
        """
        # Create test base directory
        subprocess.check_call(
            self._adb_device_sub_cmd + ["shell", "mkdir", "-p", ANDROID_HEXAGON_TEST_BASE_DIR]
        )

        # Check size of base directory and cleanup if needed
        while self._get_workspace_size() > self._workspace_max_size_mb:
            self._workspace_remove_latest()

        if not workspace_dir:
            self._workspace = str(
                ANDROID_HEXAGON_TEST_BASE_DIR
                / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            )
        else:
            self._workspace = workspace_dir

        # Upload RPC server and libraries
        subprocess.check_call(self._adb_device_sub_cmd + ["shell", "mkdir", "-p", self._workspace])

        # Create bash script
        android_bash_script_path = get_hexagon_rpc_dir() / "android_bash.sh"
        with open(get_hexagon_rpc_dir() / "android_bash.sh.template", "r") as src_f:
            if os.path.exists(android_bash_script_path):
                os.remove(android_bash_script_path)
            with open(android_bash_script_path, "w") as dest_f:
                for line in src_f.readlines():
                    if "<RPC_TRACKER_HOST>" in line:
                        line = line.replace("<RPC_TRACKER_HOST>", str(rpc_tracker_host))
                    if "<RPC_TRACKER_PORT>" in line:
                        line = line.replace("<RPC_TRACKER_PORT>", str(rpc_tracker_port))
                    if "<HEXAGON_REMOTE_DEVICE_KEY>" in line:
                        line = line.replace(
                            "<HEXAGON_REMOTE_DEVICE_KEY>", self.HEXAGON_REMOTE_DEVICE_KEY
                        )
                    if "<RPC_SERVER_PORT>" in line:
                        line = line.replace("<RPC_SERVER_PORT>", str(rpc_server_port))
                    dest_f.write(line)

        # Make shell script executable
        android_bash_stat = os.stat(android_bash_script_path)
        os.chmod(android_bash_script_path, android_bash_stat.st_mode | stat.S_IEXEC)

        # Push files
        for item in RPC_SERVER_FILES:
            src_path = get_hexagon_rpc_dir() / item
            destination = f"{self._workspace}/{item}"
            subprocess.check_call(self._adb_device_sub_cmd + ["push", src_path, destination])

        # Removed pre-defined forward/reverse rules
        subprocess.check_call(self._adb_device_sub_cmd + ["forward", "--remove-all"])
        subprocess.check_call(self._adb_device_sub_cmd + ["reverse", "--remove-all"])

        # Enable port reverse for RPC tracker
        subprocess.check_call(
            self._adb_device_sub_cmd
            + ["reverse", f"tcp:{rpc_tracker_port}", f"tcp:{rpc_tracker_port}"]
        )
        # Enable port forward for RPC server. We forward 9 ports after the rpc_server_port.
        for i in range(0, 10):
            subprocess.check_call(
                self._adb_device_sub_cmd
                + ["forward", f"tcp:{rpc_server_port+i}", f"tcp:{rpc_server_port+i}"]
            )

        # Run server and connect to tracker
        subprocess.Popen(
            self._adb_device_sub_cmd + ["shell", f"cd {self._workspace} && ./android_bash.sh"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def hexagon_setup(self):
        """Upload Hexagon artifacts on Android."""
        for item in HEXAGON_FILES:
            src_path = get_hexagon_rpc_dir() / item
            dst_path = f"{self._workspace}/{item}"
            subprocess.check_call(self._adb_device_sub_cmd + ["push", src_path, dst_path])

    def hexagon_session_setup(self, remote_kw: dict):
        """Setup Hexagon RPC Session from host to Hexagon device.

        Parameters
        ----------
        remote_kw : dict
            RPC tracker configs.
        """
        hexagon_remote_kw = dict(remote_kw)
        hexagon_remote_kw["key"] = self.HEXAGON_REMOTE_DEVICE_KEY
        self.session = Session(hexagon_remote_kw)

    def get_module(self, module_name: str):
        """Load a Hexagon TVM module, already uploaded on Android, on Hexagon and return the module.

        Parameters
        ----------
        module_name : str
            Module filename.

        Returns
        -------
        TVMModule :
            A TVM Module loaded on hexagon.
        """
        module_path = f"{self._workspace}/{module_name}"
        self._mod = self.session.load_module(module_path)
        return self._mod

    def upload(self, host_path: Union[str, pathlib.Path], remote_filename: str):
        """Upload a file to remote(Android).

        Parameters
        ----------
        host_path : Union[str, pathlib.Path]
            File path on host.

        remote_filename : str
            File name on remote(Android).
        Returns
        -------
        TVMModule :
            A TVM Module loaded on hexagon.
        """
        src_path = str(host_path)
        dst_remote_path = f"{self._workspace}/{remote_filename}"
        subprocess.check_call(self._adb_device_sub_cmd + ["push", src_path, dst_remote_path])

    def get_graph_executor(self, libmod, remote_libmod_filename: str):
        """Create a local GraphModule which consumes a remote libmod.

        Parameters
        ----------
        libmod : tvm.runtime.Module
            The module of the corresponding function.
            This library module is for remote hexagon runtime.

        remote_libmod_filename : str
            Module filename on remote. It is assumed this file lives under self._workspace path.

        Returns
        -------
        graph_module : GraphModule
            Runtime graph module that can be used to execute the graph.
        """
        self.session.__enter__()
        hexagon_mod = self.get_module(remote_libmod_filename)
        return tvm.contrib.graph_executor.create(
            libmod.get_graph_json(), hexagon_mod, self.session.device
        )

    def close(self):
        """Close RPC server on Android"""
        # Kill process childs
        subprocess.Popen(
            self._adb_device_sub_cmd + ["shell", f"pkill -P `cat {self._workspace}/rpc_pid.txt`"]
        )
        # Kill main process
        subprocess.Popen(
            self._adb_device_sub_cmd + ["shell", f"kill `cat {self._workspace}/rpc_pid.txt`"]
        )

    def _get_workspace_size(self) -> int:
        """Get workspace base directory size in MB"""
        line = subprocess.check_output(
            self._adb_device_sub_cmd + ["shell", "du", "-shm", str(ANDROID_HEXAGON_TEST_BASE_DIR)],
            encoding="utf-8",
        )
        return int(line.split("\t")[0])

    def _workspace_remove_latest(self):
        # Find oldest(lower number) directory
        latest_dir = subprocess.check_output(
            self._adb_device_sub_cmd
            + [
                "shell",
                "find",
                str(ANDROID_HEXAGON_TEST_BASE_DIR),
                "!",
                "-path",
                ".",
                "-type",
                "d",
                "|",
                "sort",
                "-n",
                "|",
                "head",
                "-1",
            ],
            encoding="utf-8",
        )
        latest_dir = latest_dir.replace("\n", "").replace("\t", "")

        subprocess.check_call(self._adb_device_sub_cmd + ["shell", "rm", "-rf", latest_dir])
