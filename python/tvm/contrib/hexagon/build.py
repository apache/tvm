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

import abc
import datetime
import multiprocessing as mp
import os
import pathlib
import signal
import socket
import stat
import random
import string
import subprocess
import tempfile
from typing import Union

import tvm
from ..._ffi import libinfo
from .session import Session


HEXAGON_RPC_LIB_DIR = os.environ.get("HEXAGON_RPC_LIB_DIR")
ANDROID_BASH_FILE_NAME = "android_bash.sh"


def _check_call_verbose(cmd, **kwargs) -> None:
    """
    Similar to subprocess.check_call(cmd), but if the exit code is non-zero
    then the raised Exception's message provides more detail, including
    the stdout/stderr provided by the subprocess.
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            encoding="UTF-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
    except subprocess.CalledProcessError as err:
        error_msg = f"{err}\nstdout:\n{err.stdout}\nstderr:\n{err.stderr}"
        raise Exception(error_msg)


def _get_hexagon_rpc_lib_dir() -> pathlib.Path:
    """Find the Hexagon API binaries.

    Returns
    -------
    pathlib.Path :
        The path to the Hexagon API directory.
    """
    global HEXAGON_RPC_LIB_DIR
    if HEXAGON_RPC_LIB_DIR is None:
        for path in libinfo.find_lib_path():
            rpc_dir = os.path.join(os.path.dirname(path), "hexagon_api_output")
            if os.path.isdir(rpc_dir):
                HEXAGON_RPC_LIB_DIR = rpc_dir
                break
        else:
            raise RuntimeError("hexagon_api binaries not found, please define HEXAGON_RPC_LIB_DIR")
    return pathlib.Path(HEXAGON_RPC_LIB_DIR)


def _get_test_directory_name() -> str:
    """Generate a time-stamped name for use as a test directory name."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    random_str = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    return f"{date_str}-{random_str}"


class HexagonLauncherRPC(metaclass=abc.ABCMeta):
    """Base class for RPC-based launchers.

    This is an abstract class intended to be a base class for specific
    implementations of RPC launchers. There are two public methods that
    each launcher needs to implement:
    - start_server
    - stop server
    and two "private" methods used in setting up the environment:
    - _copy_to_remote
    - _create_remote_directory

    The basic flow of interaction with the launcher is
        launcher = HexagonLauncher(...)
        launcher.start_server()
        with launcher.start_session() as session:
            # Do something with the session
        launcher.stop_server()
    """

    HEXAGON_REMOTE_DEVICE_KEY = "hexagon-dev"

    """Configure HexagonLauncherRPC.

    Parameters
    ----------
    rpc_info : dict
        Description of the RPC setup. Recognized keys:
            "rpc_tracker_host" : str    name of the host running the tracker (default "0.0.0.0")
            "rpc_tracker_port" : int    port number of the tracker (default: 9190)
            "rpc_server_port"  : int    port number for the RPC server to use (default 7070)
            "workspace_base"   : str    name of base test directory (default ".")
    workspace : str or patlib.Path
        The server's remote working directory. If this directory does not
        exist, it will be created. If it does exist, the servermust have
        write permissions to it.
        If this parameter is None, a subdirectory in the `workspace_base`
        directory will be created, otherwise the `workspace_base` is not
        used.
    """

    def __init__(self, rpc_info: dict, workspace: Union[str, pathlib.Path] = None):
        self._rpc_info = {
            "rpc_tracker_host": "0.0.0.0",
            "rpc_tracker_port": 9190,
            "rpc_server_port": 7070,
            "workspace_base": ".",
        }
        self._rpc_info.update(rpc_info)
        self._workspace = self._create_workspace(workspace)
        self._device_key = self.HEXAGON_REMOTE_DEVICE_KEY

    @abc.abstractmethod
    def start_server(self):
        """Start the RPC server"""
        ...

    @abc.abstractmethod
    def stop_server(self):
        """Stop the RPC server"""
        ...

    @abc.abstractmethod
    def cleanup_directory(self):
        """Cleanup working directory"""
        ...

    @abc.abstractmethod
    def _copy_to_remote(
        self, local_path: Union[str, pathlib.Path], remote_path: Union[str, pathlib.Path]
    ):
        """Copy a local file to a remote location.

        Parameters
        ----------
        local_path : str or pathlib.Path
            Path to the local file.
        remote_path : str or pathlib.Path
            Path to the remote file (to be written).
        """
        ...

    @abc.abstractmethod
    def _create_remote_directory(self, remote_path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Create a directory in the remote location.

        Parameters
        ----------
        remote_path : str or pathlib.Path
            Name of the directory to be created.

        Returns
        -------
        pathlib.Path :
            Absolute path of the remote workspace.
        """
        ...

    def _create_workspace(self, workspace: Union[str, pathlib.Path]) -> pathlib.Path:
        """Create a working directory for the server.

        Parameters
        ----------
        workspace : str or pathlib.Path or NoneType
            Name of the directory to create. If None, a new name is constructed
            using workspace_base.

        Returns
        -------
        pathlib.Path :
            Created workspace.
        """
        if not workspace:
            base_dir = self._rpc_info["workspace_base"]
            workspace = os.path.join(base_dir, _get_test_directory_name())
        return self._create_remote_directory(workspace)

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
        assert self._workspace
        remote_file_path = self._workspace / remote_filename
        self._copy_to_remote(local_path, str(remote_file_path))
        return remote_file_path

    def start_session(self, session_name: str = "hexagon-rpc") -> Session:
        """Connect to the RPC server.

        Parameters
        ----------
        session_name : str
            RPC session name.

        Returns
        -------
        Session :
            The session object.
        """
        hexagon_remote_kw = {
            "host": self._rpc_info["rpc_tracker_host"],
            "port": self._rpc_info["rpc_tracker_port"],
            "priority": 0,
            "timeout": 0,
            "key": self._device_key,
        }
        return Session(self, hexagon_remote_kw, session_name=session_name)

    def load_module(self, module: Union[str, pathlib.Path, tvm.runtime.Module], session: Session):
        """Load TVM module.

        Parameters
        ----------
        module : Union[str, pathlib.Path, tvm.runtime.Module]

            The module to load.  If `module` is a
            `tvm.runtime.Module`, it will be uploaded to the remote
            session and loaded.

            If the object passed is a string or pathlib.Path, it must
            be a full path in the remote system.

        session : Session

            Remote session. The session must be established (via __enter__)
            prior to calling this function.

        Returns
        -------
        TVMModule :
            TVM module object.

        """
        return session.load_module(module)

    def get_graph_executor(
        self,
        graph_json: str,
        module: Union[str, pathlib.Path, tvm.runtime.Module],
        session: Session,
    ):
        """Create a local GraphModule which consumes a remote libmod.

        Parameters
        ----------
        graph_json : str
            The string with the graph JSON.
        module : Union[str, pathlib.Path, tvm.runtime.Module]

            The module to load.  If `module` is a
            `tvm.runtime.Module`, it will be uploaded to the remote
            session and loaded.

            If the object passed is a string or pathlib.Path, it must
            be a full path in the remote system.
        session : Session
            Remote session. The session must be established (via __enter__)
            prior to calling this function.

        Returns
        -------
        GraphModule :
            Runtime graph module that can be used to execute the graph.
        """
        return session.get_graph_executor(graph_json, module)

    def get_graph_debug_executor(
        self,
        graph_json: str,
        module: Union[str, pathlib.Path, tvm.runtime.Module],
        session: Session,
        dump_root: Union[str, pathlib.Path] = None,
    ):
        """Create a local GraphModuleDebug which consumes a remote libmod.

        Parameters
        ----------
        graph_json : str
            The string with the graph JSON.
        module : Union[str, pathlib.Path, tvm.runtime.Module]

            The module to load.  If `module` is a
            `tvm.runtime.Module`, it will be uploaded to the remote
            session and loaded.

            If the object passed is a string or pathlib.Path, it must
            be a full path in the remote system.
        session : Session
            Remote session. The session must be established (via __enter__)
            prior to calling this function.

        Returns
        -------
        GraphModuleDebug :
            Runtime debug graph module that can be used to debug the graph.
        """
        return session.get_graph_debug_executor(graph_json, module, dump_root=dump_root)


class HexagonLauncherAndroid(HexagonLauncherRPC):
    """Hexagon Launcher for Android."""

    ANDROID_HEXAGON_TEST_BASE_DIR = pathlib.Path("/data/local/tmp/hexagon_test")
    ANDROID_HEXAGON_RPC_FILES = [
        "libhexagon_rpc_skel.so",
        "libtvm_runtime.so",
        "tvm_rpc_android",
    ]

    def __init__(
        self, serial_number: str, rpc_info: dict, workspace: Union[str, pathlib.Path] = None
    ):
        """Configure a new HexagonLauncherAndroid

        Parameters
        ----------
        serial_number : str
            Android device serial number.
        rpc_info : dict
            Same as in HexagonLauncherRPC, except if the "workspace_base"
            key is not present or is None, ANDROID_HEXAGON_TEST_BASE_DIR
            is used as the base directory.
        workspace : str or pathlib.Path, optional
            Test workspace path on android.
        """
        if not rpc_info.get("workspace_base"):
            rpc_info["workspace_base"] = self.ANDROID_HEXAGON_TEST_BASE_DIR
        self._serial_number = serial_number
        adb_socket = rpc_info["adb_server_socket"] if rpc_info["adb_server_socket"] else "tcp:5037"
        self._adb_device_sub_cmd = ["adb", "-L", adb_socket, "-s", self._serial_number]
        self.forwarded_ports_ = []

        super(HexagonLauncherAndroid, self).__init__(rpc_info, workspace)

    def _copy_to_remote(
        self, local_path: Union[str, pathlib.Path], remote_path: Union[str, pathlib.Path]
    ):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        _check_call_verbose(self._adb_device_sub_cmd + ["push", str(local_path), str(remote_path)])

    def _create_remote_directory(self, remote_path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        _check_call_verbose(self._adb_device_sub_cmd + ["shell", "mkdir", "-p", str(remote_path)])
        return pathlib.Path(remote_path)

    def _copy_binaries(self):
        """Upload Android server binaries."""

        # Create bash script
        with open(_get_hexagon_rpc_lib_dir() / f"{ANDROID_BASH_FILE_NAME}.template", "r") as src_f:
            with tempfile.TemporaryDirectory() as temp_dir:
                android_bash_script_path = pathlib.Path(temp_dir) / ANDROID_BASH_FILE_NAME
                with open(android_bash_script_path, "w") as dest_f:
                    for line in src_f.readlines():
                        if "<RPC_TRACKER_HOST>" in line:
                            line = line.replace(
                                "<RPC_TRACKER_HOST>", str(self._rpc_info["rpc_tracker_host"])
                            )
                        if "<RPC_TRACKER_PORT>" in line:
                            line = line.replace(
                                "<RPC_TRACKER_PORT>", str(self._rpc_info["rpc_tracker_port"])
                            )
                        if "<HEXAGON_REMOTE_DEVICE_KEY>" in line:
                            line = line.replace("<HEXAGON_REMOTE_DEVICE_KEY>", self._device_key)
                        if "<RPC_SERVER_PORT>" in line:
                            line = line.replace(
                                "<RPC_SERVER_PORT>", str(self._rpc_info["rpc_server_port"])
                            )
                        dest_f.write(line)

                # Make shell script executable
                android_bash_stat = os.stat(android_bash_script_path)
                os.chmod(android_bash_script_path, android_bash_stat.st_mode | stat.S_IEXEC)
                self._copy_to_remote(
                    android_bash_script_path, self._workspace / android_bash_script_path.name
                )

        # Push files
        lib_dir = _get_hexagon_rpc_lib_dir()
        for item in self.ANDROID_HEXAGON_RPC_FILES:
            self._copy_to_remote(lib_dir / item, self._workspace / item)

    def _process_forwarded_ports(self):
        forwarded_ports = subprocess.check_output(self._adb_device_sub_cmd + ["forward", "--list"])
        existing_forwards = []
        for forward in str(forwarded_ports).split("\\n"):
            entry = forward.split()
            if len(entry) == 3:
                _, local, _ = entry
                existing_forwards.append(int(local.strip("tcp:")))
        return existing_forwards

    def _forward_ports(self, rpc_server_port, existing_forwards):
        # Enable port forward for RPC server. We forward the first ten open ports
        # starting from the rpc_server_port
        port = rpc_server_port
        while len(self.forwarded_ports_) < 10:
            if port not in existing_forwards and not _is_port_in_use(port):
                _check_call_verbose(
                    self._adb_device_sub_cmd + ["forward", f"tcp:{port}", f"tcp:{port}"]
                )
                self.forwarded_ports_.append(port)
            port += 1

    def _reverse_ports(self, rpc_tracker_port):
        _check_call_verbose(
            self._adb_device_sub_cmd
            + ["reverse", f"tcp:{rpc_tracker_port}", f"tcp:{rpc_tracker_port}"]
        )

    def _run_server_script(self):
        """Setup the ADB connection and execute the server script."""

        # Collect any existing adb port forwarding to avoid duplication
        # with another running process
        existing_forwards = self._process_forwarded_ports()
        # Enable port reverse for RPC tracker
        rpc_tracker_port = self._rpc_info["rpc_tracker_port"]
        rpc_server_port = self._rpc_info["rpc_server_port"]

        self._reverse_ports(rpc_tracker_port)
        self._forward_ports(rpc_server_port, existing_forwards)

        # Run server and connect to tracker
        subprocess.Popen(
            self._adb_device_sub_cmd
            + ["shell", f"cd {self._workspace} && ./{ANDROID_BASH_FILE_NAME}"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _cleanup_port_forwarding(self):
        # Removed pre-defined forward/reverse rules
        rpc_tracker_port = self._rpc_info["rpc_tracker_port"]
        _check_call_verbose(
            self._adb_device_sub_cmd + ["reverse", "--remove", f"tcp:{rpc_tracker_port}"]
        )
        for port in self.forwarded_ports_:
            _check_call_verbose(self._adb_device_sub_cmd + ["forward", "--remove", f"tcp:{port}"])

    def _terminate_remote(self):
        # Send interupt to main and child processes
        subprocess.Popen(
            self._adb_device_sub_cmd
            + ["shell", f"pkill -l sigint -P `cat {self._workspace}/rpc_pid.txt`"]
        )
        subprocess.Popen(
            self._adb_device_sub_cmd
            + ["shell", f"kill -s sigint `cat {self._workspace}/rpc_pid.txt`"]
        )
        # Wait for processes to destruct cleanly after receiving the intrupt
        subprocess.Popen(self._adb_device_sub_cmd + ["shell", "sleep", "0.1s"])
        # Kill process children
        subprocess.Popen(
            self._adb_device_sub_cmd + ["shell", f"pkill -P `cat {self._workspace}/rpc_pid.txt`"]
        )
        # Kill main process
        subprocess.Popen(
            self._adb_device_sub_cmd + ["shell", f"kill `cat {self._workspace}/rpc_pid.txt`"]
        )

    def cleanup_directory(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        subprocess.Popen(self._adb_device_sub_cmd + ["shell", f"rm -rf {self._workspace}"])

    def start_server(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        self._copy_binaries()
        self._run_server_script()

    def stop_server(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        self._cleanup_port_forwarding()
        self._terminate_remote()
        self.cleanup_directory()


class HexagonLauncherSimulator(HexagonLauncherRPC):
    """Hexagon Launcher for Hexagon simulator."""

    SIMULATOR_HEXAGON_RPC_FILES = ["tvm_rpc_x86", "libhexagon_rpc_sim.so"]

    def __init__(self, rpc_info: dict, workspace: Union[str, pathlib.Path] = None):
        """Configure a new HexagonLauncherSimulator

        Parameters are same as for HexagonLauncherRPC.
        """
        super(HexagonLauncherSimulator, self).__init__(rpc_info, workspace)

        self._toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
        if not self._toolchain:
            raise RuntimeError("Please set HEXAGON_TOOLCHAIN env variable")
        self._serial_number = "simulator"

    def _copy_to_remote(
        self, local_path: Union[str, pathlib.Path], remote_path: Union[str, pathlib.Path]
    ):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        _check_call_verbose(["cp", str(local_path), str(remote_path)])

    def _create_remote_directory(self, remote_path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        _check_call_verbose(["mkdir", "-p", str(remote_path)])
        return pathlib.Path(os.path.abspath(remote_path))

    def _copy_libcxx(self, dest_dir: Union[str, pathlib.Path]):
        """Copy libc++ libraries to the remote workspace."""
        # Copy the v68 versions, since we don't have target information.
        # The v68 ones should work everywhere on v68+.
        lib_dir = os.path.join(self._toolchain, "target/hexagon/lib/v68/G0/pic")

        libcxx_files = []
        for entry in os.scandir(lib_dir):
            if entry.is_dir() or entry.name.find(".so") == -1:
                continue
            if entry.name.startswith("libc++"):
                libcxx_files.append(entry.name)

        # Use tar to preserve the symbolic links. Libc++ libraries use the
        # typical .so versioning, so that libc++.so may be a symlink to
        # something else. Also, shared libraries using libc++ could be
        # directly linked against some version, e.g. libc++.so.1, so make
        # sure that all files are copied over. The preservation of symbolic
        # links is to save disk space.
        tar_in = f"tar -cf - -C {lib_dir} " + " ".join(libcxx_files)
        tar_out = f"tar -xf - -C {str(dest_dir)}"
        _check_call_verbose(tar_in + " | " + tar_out, shell=True)

    def start_server(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        # Copy binaries
        lib_dir = _get_hexagon_rpc_lib_dir()
        for item in self.SIMULATOR_HEXAGON_RPC_FILES:
            self._copy_to_remote(lib_dir / item, self._workspace / item)
        # Copy libc++ from the toolchain to the workspace
        self._copy_libcxx(self._workspace)
        self._device_key = self.HEXAGON_REMOTE_DEVICE_KEY + "." + str(os.getpid())

        rpc_tracker_host = self._rpc_info["rpc_tracker_host"]
        rpc_tracker_port = self._rpc_info["rpc_tracker_port"]
        rpc_server_port = self._rpc_info["rpc_server_port"]
        server_exe = os.path.join(".", "tvm_rpc_x86")

        args = [
            "server",
            f"--tracker={rpc_tracker_host}:{rpc_tracker_port}",
            f"--port={rpc_server_port}",
            f"--key={self._device_key}",
            "--timeout=0",
        ]

        # pylint: disable=unused-argument
        def _terminate_handler(self, signum, *rest):
            # Terminate the Popen'ed (sub)process.
            os.kill(self._subprocess_pid, signal.SIGTERM)

        def _start(self):
            # This function will be running in a new process. It will start the RPC
            # (x86) server as a subprocess of itself.
            log_out = self._workspace / "stdout.txt"
            log_err = self._workspace / "stderr.txt"
            # Intercept the TERM signal so we can also terminate the subprocess.
            signal.signal(signal.SIGTERM, lambda *a: _terminate_handler(self, *a))

            with open(log_out, "w") as out, open(log_err, "w") as err:
                p = subprocess.Popen(
                    [server_exe, *args], stdout=out, stderr=err, cwd=self._workspace
                )
                # Insert the pid of the subprocess in the self object.
                self._subprocess_pid = p.pid
                p.wait()

        self._server_process = mp.Process(target=lambda *a: _start(self, *a))
        self._server_process.start()

    def cleanup_directory(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""

    def stop_server(self):
        """Abstract method implementation. See description in HexagonLauncherRPC."""
        self._server_process.terminate()


# https://stackoverflow.com/a/52872579/2689797
def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# pylint: disable=invalid-name
def HexagonLauncher(serial_number: str, rpc_info: dict, workspace: Union[str, pathlib.Path] = None):
    if serial_number == "simulator":
        return HexagonLauncherSimulator(rpc_info, workspace)
    return HexagonLauncherAndroid(serial_number, rpc_info, workspace)
