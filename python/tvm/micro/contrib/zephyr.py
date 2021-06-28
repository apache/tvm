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

"""Defines a compiler integration that uses an externally-supplied Zephyr project."""

import collections
import copy
import logging
import multiprocessing
import os
import pathlib
import re
import tempfile
import textwrap
import shlex
import shutil
import subprocess
import sys
import threading
import queue
import enum

import yaml

import tvm.micro
from . import base
from .. import compiler
from .. import debugger
from ..transport import debug
from ..transport import file_descriptor

from ..transport import serial
from ..transport import Transport, TransportClosedError, TransportTimeouts
from ..transport import wakeup


_LOG = logging.getLogger(__name__)


class SubprocessEnv(object):
    def __init__(self, default_overrides):
        self.default_overrides = default_overrides

    def run(self, cmd, **kw):
        env = dict(os.environ)
        for k, v in self.default_overrides.items():
            env[k] = v

        return subprocess.check_output(cmd, env=env, **kw, universal_newlines=True)


class ProjectNotFoundError(Exception):
    """Raised when the project_dir supplied to ZephyrCompiler does not exist."""


class FlashRunnerNotSupported(Exception):
    """Raised when the FLASH_RUNNER for a project isn't supported by this Zephyr adapter."""


class ZephyrCompiler(tvm.micro.Compiler):
    """A Compiler instance that builds against a pre-existing zephyr project."""

    def __init__(
        self,
        project_dir=None,
        board=None,
        west_cmd=None,
        zephyr_base=None,
        zephyr_toolchain_variant=None,
        env_vars=None,
    ):
        """Configure the compiler for use.

        Parameters
        ----------
        project_dir : str
            Path to the pre-existing Zephyr project.
        board : str
            Name of the Zephyr board to build for (i.e. passed to `west build -b`)
        west_cmd : Optional[list]
            If given, argv that invoke the west build tool. Used only for flashing.
        zephyr_base : Optional[str]
            If given, path to Zephyr, as would normally be present in the ZEPHYR_BASE environment
            variable. If not given, consults this environment variable. This value must be set in
            one of those two places.
        zephyr_toolchain_variant: Optional[str]
            If given, overrides the toolchain used by Zephyr. If not given, uses the default
            zephyr toolchain. When running on OS X outside of docker, you need to specify this.
        env_vars : Optional[Dict[str,str]]
            If given, additional environment variables present when invoking west, cmake, or make.
        """
        self._project_dir = project_dir
        if not os.path.exists(project_dir):
            # Raise this error instead of a potentially-more-cryptic compiler error due to a missing
            # prj.conf.
            raise ProjectNotFoundError(
                f"project_dir supplied to ZephyrCompiler does not exist: {project_dir}"
            )

        self._qemu = "qemu" in board

        # For Zephyr boards that run emulated by default but don't have the prefix "qemu_" in their
        # board names, a suffix "-qemu" is added by users of microTVM when specifying the board
        # name to inform that the QEMU transporter must be used just like for the boards with
        # the prefix. Zephyr does not recognize the suffix, so we trim it off before passing it.
        if "-qemu" in board:
            board = board.replace("-qemu", "")

        self._board = board

        if west_cmd is None:
            self._west_cmd = [sys.executable, "-mwest.app.main"]
        elif isinstance(west_cmd, str):
            self._west_cmd = [west_cmd]
        elif isinstance(west_cmd, list):
            self._west_cmd = west_cmd
        else:
            raise TypeError("west_cmd: expected string, list, or None; got %r" % (west_cmd,))

        env = {}
        if zephyr_toolchain_variant is not None:
            env["ZEPHYR_TOOLCHAIN_VARIANT"] = zephyr_toolchain_variant

        self._zephyr_base = zephyr_base or os.environ["ZEPHYR_BASE"]
        assert (
            self._zephyr_base is not None
        ), f"Must specify zephyr_base=, or ZEPHYR_BASE must be in environment variables"
        env["ZEPHYR_BASE"] = self._zephyr_base

        if env_vars:
            env.update(env_vars)

        self._subprocess_env = SubprocessEnv(env)

    OPT_KEY_TO_CMAKE_DEFINE = {
        "cflags": "CFLAGS",
        "ccflags": "CXXFLAGS",
        "ldflags": "LDFLAGS",
    }

    @classmethod
    def _options_to_cmake_args(cls, options):
        args = []
        for key, define in cls.OPT_KEY_TO_CMAKE_DEFINE.items():
            if key in options:
                quoted_opts = [shlex.quote(o).replace(";", "\\;") for o in options[key]]
                args.append(f'-DEXTRA_{define}={" ".join(quoted_opts)}')

        if "cmake_args" in options:
            args.extend(options["cmake_args"])

        return args

    def library(self, output, sources, options=None):
        project_name = os.path.basename(output)
        if project_name.startswith("lib"):
            project_name = project_name[3:]

        lib_prj_conf = os.path.join(output, "prj.conf")
        if self._project_dir is not None:
            project_dir_conf = os.path.join(self._project_dir, "prj.conf")
            if os.path.exists(project_dir_conf):
                shutil.copy(project_dir_conf, lib_prj_conf)

            # Copy board-specific Zephyr config file from the project_dir to
            # the build lib dir so board-specific configs can be found and used by
            # Zephyr's build system in conjunction with the generic prj.conf configs.
            board_conf = os.path.join("boards", self._board + ".conf")
            project_dir_board_conf = os.path.join(self._project_dir, board_conf)
            if os.path.exists(project_dir_board_conf):
                os.mkdir(os.path.join(output, "boards"))
                lib_dir_board_conf = os.path.join(output, board_conf)
                shutil.copy(project_dir_board_conf, lib_dir_board_conf)

        else:
            with open(lib_prj_conf, "w") as prj_conf_f:
                prj_conf_f.write("CONFIG_CPLUSPLUS=y\n")

        cmakelists_path = os.path.join(output, "CMakeLists.txt")
        with open(cmakelists_path, "w") as cmake_f:
            sources = " ".join(f'"{o}"' for o in sources)
            cmake_f.write(
                textwrap.dedent(
                    f"""\
                cmake_minimum_required(VERSION 3.13.1)

                find_package(Zephyr HINTS $ENV{{ZEPHYR_BASE}})
                project({project_name}_prj)
                target_sources(app PRIVATE)
                zephyr_library_named({project_name})
                target_sources({project_name} PRIVATE {sources})
                target_sources(app PRIVATE main.c)
                target_link_libraries(app PUBLIC {project_name})
                """
                )
            )
            if "include_dirs" in options:
                cmake_f.write(
                    f"target_include_directories({project_name} PRIVATE "
                    f'{" ".join(os.path.abspath(d) for d in options["include_dirs"])})\n'
                )

        with open(os.path.join(output, "main.c"), "w"):
            pass

        # expected not to exist after populate_tvm_libs
        build_dir = os.path.join(output, "__tvm_build")
        os.mkdir(build_dir)
        self._subprocess_env.run(
            ["cmake", "..", f"-DBOARD={self._board}"] + self._options_to_cmake_args(options),
            cwd=build_dir,
        )
        num_cpus = multiprocessing.cpu_count()
        self._subprocess_env.run(
            ["make", f"-j{num_cpus}", "VERBOSE=1", project_name], cwd=build_dir
        )
        return tvm.micro.MicroLibrary(build_dir, [f"lib{project_name}.a"])

    def _print_make_statistics(self, output):
        output = output.splitlines()
        lines = iter(output)
        for line in lines:
            if line.startswith("Memory region"):
                # print statistics header
                _LOG.info(line)
                _LOG.info("--------------------- ---------- ------------ ---------")
                line = next(lines)
                # while there is a region print it
                try:
                    while ":" in line:
                        _LOG.info(line)
                        line = next(lines)
                    else:
                        break
                except StopIteration:
                    pass

    def binary(self, output, objects, options=None, link_main=True, main_options=None):
        assert link_main, "Must pass link_main=True"
        assert self._project_dir is not None, "Must supply project_dir= to build binaries"

        copied_libs = base.populate_tvm_objs(self._project_dir, objects)

        # expected not to exist after populate_tvm_objs
        cmake_args = [
            "cmake",
            os.path.abspath(self._project_dir),
            f"-DBOARD={self._board}",
        ] + self._options_to_cmake_args(options)
        if "include_dirs" in options:
            cmake_args.append(
                "-DTVM_INCLUDE_DIRS="
                f'{";".join(os.path.abspath(d) for d in options["include_dirs"])}'
            )
        cmake_args.append(f'-DTVM_LIBS={";".join(copied_libs)}')
        self._subprocess_env.run(cmake_args, cwd=output)

        make_output = self._subprocess_env.run(["make"], cwd=output)

        self._print_make_statistics(make_output)

        return tvm.micro.MicroBinary(
            output,
            binary_file=os.path.join("zephyr", "zephyr.elf"),
            debug_files=[os.path.join("zephyr", "zephyr.elf")],
            labelled_files={
                "cmake_cache": ["CMakeCache.txt"],
                "device_tree": [os.path.join("zephyr", "zephyr.dts")],
            },
            immobile=bool(self._qemu),
        )

    @property
    def flasher_factory(self):
        return compiler.FlasherFactory(
            ZephyrFlasher,
            (
                self._board,
                self._qemu,
            ),
            dict(
                zephyr_base=self._zephyr_base,
                project_dir=self._project_dir,
                subprocess_env=self._subprocess_env.default_overrides,
                west_cmd=self._west_cmd,
            ),
        )


CACHE_ENTRY_RE = re.compile(r"(?P<name>[^:]+):(?P<type>[^=]+)=(?P<value>.*)")


CMAKE_BOOL_MAP = dict(
    [(k, True) for k in ("1", "ON", "YES", "TRUE", "Y")]
    + [(k, False) for k in ("0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "")]
)


def read_cmake_cache(file_name):
    """Read a CMakeCache.txt-like file and return a dictionary of values."""
    entries = collections.OrderedDict()
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            m = CACHE_ENTRY_RE.match(line.rstrip("\n"))
            if not m:
                continue

            if m.group("type") == "BOOL":
                value = CMAKE_BOOL_MAP[m.group("value").upper()]
            else:
                value = m.group("value")

            entries[m.group("name")] = value

    return entries


class BoardError(Exception):
    """Raised when an attached board cannot be opened (i.e. missing /dev nodes, etc)."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the board= given to ZephyrCompiler."""


class ZephyrFlasher(tvm.micro.compiler.Flasher):
    """A Flasher implementation that delegates to Zephyr/west."""

    def __init__(
        self,
        board,
        qemu,
        zephyr_base=None,
        project_dir=None,
        subprocess_env=None,
        nrfjprog_snr=None,
        openocd_serial=None,
        flash_args=None,
        debug_rpc_session=None,
        serial_timeouts=None,
        west_cmd=None,
    ):
        zephyr_base = zephyr_base or os.environ["ZEPHYR_BASE"]
        sys.path.insert(0, os.path.join(zephyr_base, "scripts", "dts"))
        try:
            import dtlib  # pylint: disable=import-outside-toplevel

            self._dtlib = dtlib
        finally:
            sys.path.pop(0)

        self._board = board
        self._qemu = qemu
        self._zephyr_base = zephyr_base
        self._project_dir = project_dir
        self._west_cmd = west_cmd
        self._flash_args = flash_args
        self._openocd_serial = openocd_serial
        self._autodetected_openocd_serial = None
        self._subprocess_env = SubprocessEnv(subprocess_env)
        self._debug_rpc_session = debug_rpc_session
        self._nrfjprog_snr = nrfjprog_snr
        self._serial_timeouts = serial_timeouts

    def _get_nrf_device_args(self):
        nrfjprog_args = ["nrfjprog", "--ids"]
        nrfjprog_ids = subprocess.check_output(nrfjprog_args, encoding="utf-8")
        if not nrfjprog_ids.strip("\n"):
            raise BoardAutodetectFailed(
                f'No attached boards recognized by {" ".join(nrfjprog_args)}'
            )

        boards = nrfjprog_ids.split("\n")[:-1]
        if len(boards) > 1:
            if self._nrfjprog_snr is None:
                raise BoardError(
                    "Multiple boards connected; specify one with nrfjprog_snr=: "
                    f'{", ".join(boards)}'
                )

            if str(self._nrfjprog_snr) not in boards:
                raise BoardError(
                    f"nrfjprog_snr ({self._nrfjprog_snr}) not found in {nrfjprog_args}: {boards}"
                )

            return ["--snr", str(self._nrfjprog_snr)]

        if not boards:
            return []

        return ["--snr", boards[0]]

    # kwargs passed to usb.core.find to find attached boards for the openocd flash runner.
    BOARD_USB_FIND_KW = {
        "nucleo_f746zg": {"idVendor": 0x0483, "idProduct": 0x374B},
        "stm32f746g_disco": {"idVendor": 0x0483, "idProduct": 0x374B},
    }

    def openocd_serial(self, cmake_entries):
        """Find the serial port to use for a board with OpenOCD flash strategy."""
        if self._openocd_serial is not None:
            return self._openocd_serial

        if self._autodetected_openocd_serial is None:
            import usb  # pylint: disable=import-outside-toplevel

            find_kw = self.BOARD_USB_FIND_KW[cmake_entries["BOARD"]]
            boards = usb.core.find(find_all=True, **find_kw)
            serials = []
            for b in boards:
                serials.append(b.serial_number)

            if len(serials) == 0:
                raise BoardAutodetectFailed(f"No attached USB devices matching: {find_kw!r}")
            serials.sort()

            self._autodetected_openocd_serial = serials[0]
            _LOG.debug("zephyr openocd driver: autodetected serial %s", serials[0])

        return self._autodetected_openocd_serial

    def _get_openocd_device_args(self, cmake_entries):
        return ["--serial", self.openocd_serial(cmake_entries)]

    @classmethod
    def _get_flash_runner(cls, cmake_entries):
        flash_runner = cmake_entries.get("ZEPHYR_BOARD_FLASH_RUNNER")
        if flash_runner is not None:
            return flash_runner

        with open(cmake_entries["ZEPHYR_RUNNERS_YAML"]) as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
        return doc["flash-runner"]

    def _get_device_args(self, cmake_entries):
        flash_runner = self._get_flash_runner(cmake_entries)

        if flash_runner == "nrfjprog":
            return self._get_nrf_device_args()
        if flash_runner == "openocd":
            return self._get_openocd_device_args(cmake_entries)

        raise BoardError(
            f"Don't know how to find serial terminal for board {cmake_entries['BOARD']} with flash "
            f"runner {flash_runner}"
        )

    def _zephyr_transport(self, micro_binary):
        qemu_debugger = None
        if self._debug_rpc_session:
            qemu_debugger = debugger.RpcDebugger(
                self._debug_rpc_session,
                debugger.DebuggerFactory(
                    QemuGdbDebugger,
                    (micro_binary.abspath(micro_binary.debug_files[0]),),
                    {},
                ),
            )

        return ZephyrQemuTransport(
            micro_binary.base_dir, startup_timeout_sec=30.0, qemu_debugger=qemu_debugger
        )

    def flash(self, micro_binary):
        if self._qemu:
            return self._zephyr_transport(micro_binary)

        cmake_cache_path = micro_binary.abspath(micro_binary.labelled_files["cmake_cache"][0])
        cmake_entries = read_cmake_cache(cmake_cache_path)

        build_dir = os.path.dirname(cmake_cache_path)

        # The nRF5340DK requires an additional `nrfjprog --recover` before each flash cycle.
        # This is because readback protection is enabled by default when this device is flashed.
        # Otherwise, flashing may fail with an error such as the following:
        #  ERROR: The operation attempted is unavailable due to readback protection in
        #  ERROR: your device. Please use --recover to unlock the device.
        if (
            self._board.startswith("nrf5340dk")
            and self._get_flash_runner(cmake_entries) == "nrfjprog"
        ):
            recover_args = ["nrfjprog", "--recover"]
            recover_args.extend(self._get_nrf_device_args())
            self._subprocess_env.run(recover_args, cwd=build_dir)

        west_args = (
            self._west_cmd
            + ["flash", "--build-dir", build_dir, "--skip-rebuild"]
            + self._get_device_args(cmake_entries)
        )
        if self._flash_args is not None:
            west_args.extend(self._flash_args)
        self._subprocess_env.run(west_args, cwd=build_dir)

        return self.transport(micro_binary)

    def _find_nrf_serial_port(self, cmake_entries):
        com_ports = subprocess.check_output(
            ["nrfjprog", "--com"] + self._get_device_args(cmake_entries), encoding="utf-8"
        )
        ports_by_vcom = {}
        for line in com_ports.split("\n")[:-1]:
            parts = line.split()
            ports_by_vcom[parts[2]] = parts[1]

        return {"port_path": ports_by_vcom["VCOM2"]}

    def _find_openocd_serial_port(self, cmake_entries):
        return {"grep": self.openocd_serial(cmake_entries)}

    def _find_serial_port(self, micro_binary):
        cmake_entries = read_cmake_cache(
            micro_binary.abspath(micro_binary.labelled_files["cmake_cache"][0])
        )
        flash_runner = self._get_flash_runner(cmake_entries)

        if flash_runner == "nrfjprog":
            return self._find_nrf_serial_port(cmake_entries)

        if flash_runner == "openocd":
            return self._find_openocd_serial_port(cmake_entries)

        raise FlashRunnerNotSupported(
            f"Don't know how to deduce serial port for flash runner {flash_runner}"
        )

    def transport(self, micro_binary):
        """Instantiate the transport for use with non-QEMU Zephyr."""
        dt_inst = self._dtlib.DT(
            micro_binary.abspath(micro_binary.labelled_files["device_tree"][0])
        )
        uart_baud = (
            dt_inst.get_node("/chosen")
            .props["zephyr,console"]
            .to_path()
            .props["current-speed"]
            .to_num()
        )
        _LOG.debug("zephyr transport: found UART baudrate from devicetree: %d", uart_baud)

        port_kwargs = self._find_serial_port(micro_binary)
        serial_transport = serial.SerialTransport(
            timeouts=self._serial_timeouts, baudrate=uart_baud, **port_kwargs
        )
        if self._debug_rpc_session is None:
            return serial_transport

        return debug.DebugWrapperTransport(
            debugger.RpcDebugger(
                self._debug_rpc_session,
                debugger.DebuggerFactory(
                    ZephyrDebugger,
                    (
                        " ".join(shlex.quote(x) for x in self._west_cmd),
                        os.path.dirname(micro_binary.abspath(micro_binary.label("cmake_cache")[0])),
                        micro_binary.abspath(micro_binary.debug_files[0]),
                        self._zephyr_base,
                    ),
                    {},
                ),
            ),
            serial_transport,
        )


class QemuGdbDebugger(debugger.GdbDebugger):
    def __init__(self, elf_file):
        super(QemuGdbDebugger, self).__init__()
        self._elf_file = elf_file

    def popen_kwargs(self):
        # expect self._elf file to follow the form .../zephyr/zephyr.elf
        cmake_cache_path = pathlib.Path(self._elf_file).parent.parent / "CMakeCache.txt"
        cmake_cache = read_cmake_cache(cmake_cache_path)
        return {
            "args": [
                cmake_cache["CMAKE_GDB"],
                "-ex",
                "target remote localhost:1234",
                "-ex",
                f"file {self._elf_file}",
            ],
        }


class QemuStartupFailureError(Exception):
    """Raised when the qemu pipe is not present within startup_timeout_sec."""


class QemuFdTransport(file_descriptor.FdTransport):
    """An FdTransport subclass that escapes written data to accommodate the QEMU monitor.

    It's supposedly possible to disable the monitor, but Zephyr controls most of the command-line
    arguments for QEMU and there are too many options which implictly enable the monitor, so this
    approach seems more robust.
    """

    def write_monitor_quit(self):
        file_descriptor.FdTransport.write(self, b"\x01x", 1.0)

    def close(self):
        file_descriptor.FdTransport.close(self)

    def timeouts(self):
        assert False, "should not get here"

    def write(self, data, timeout_sec):
        """Write data, escaping for QEMU monitor."""
        to_write = bytearray()
        escape_pos = []
        for i, b in enumerate(data):
            if b == 0x01:
                to_write.append(b)
                escape_pos.append(i)
            to_write.append(b)

        num_written = file_descriptor.FdTransport.write(self, to_write, timeout_sec)
        num_written -= sum(1 if x < num_written else 0 for x in escape_pos)
        return num_written


class ZephyrQemuMakeResult(enum.Enum):
    QEMU_STARTED = "qemu_started"
    MAKE_FAILED = "make_failed"
    EOF = "eof"


class ZephyrQemuTransport(Transport):
    """The user-facing Zephyr QEMU transport class."""

    def __init__(self, base_dir, startup_timeout_sec=5.0, qemu_debugger=None, **kwargs):
        self.base_dir = base_dir
        self.startup_timeout_sec = startup_timeout_sec
        self.kwargs = kwargs
        self.proc = None
        self.fd_transport = None
        self.pipe_dir = None
        self.qemu_debugger = qemu_debugger
        self._queue = queue.Queue()

    def timeouts(self):
        return TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=self.startup_timeout_sec,
            session_established_timeout_sec=5.0 if self.qemu_debugger is None else 0,
        )

    def open(self):
        self.pipe_dir = tempfile.mkdtemp()
        self.pipe = os.path.join(self.pipe_dir, "fifo")
        self.write_pipe = os.path.join(self.pipe_dir, "fifo.in")
        self.read_pipe = os.path.join(self.pipe_dir, "fifo.out")

        os.mkfifo(self.write_pipe)
        os.mkfifo(self.read_pipe)
        if self.qemu_debugger is not None:
            if "env" in self.kwargs:
                self.kwargs["env"] = copy.copy(self.kwargs["env"])
            else:
                self.kwargs["env"] = os.environ.copy()

            self.kwargs["env"]["TVM_QEMU_DEBUG"] = "1"

        self.proc = subprocess.Popen(
            ["make", "run", f"QEMU_PIPE={self.pipe}"],
            cwd=self.base_dir,
            **self.kwargs,
            stdout=subprocess.PIPE,
        )
        try:
            self._wait_for_qemu()
        except Exception as error:
            raise error

        if self.qemu_debugger is not None:
            self.qemu_debugger.start()

        # NOTE: although each pipe is unidirectional, open both as RDWR to work around a select
        # limitation on linux. Without this, non-blocking I/O can't use timeouts because named
        # FIFO are always considered ready to read when no one has opened them for writing.
        self.fd_transport = wakeup.WakeupTransport(
            QemuFdTransport(
                os.open(self.read_pipe, os.O_RDWR | os.O_NONBLOCK),
                os.open(self.write_pipe, os.O_RDWR | os.O_NONBLOCK),
                self.timeouts(),
            ),
            b"\xfe\xff\xfd\x03\0\0\0\0\0\x02" b"fw",
        )
        self.fd_transport.open()

    def close(self):
        if self.qemu_debugger is not None:
            self.qemu_debugger.stop()

        if self.fd_transport is not None:
            self.fd_transport.child_transport.write_monitor_quit()
            self.proc.wait()
            self.fd_transport.close()
            self.fd_transport = None

        if self.proc is not None:
            self.proc = None

        if self.pipe_dir is not None:
            shutil.rmtree(self.pipe_dir)
            self.pipe_dir = None

    def read(self, n, timeout_sec):
        if self.fd_transport is None:
            raise TransportClosedError()
        return self.fd_transport.read(n, timeout_sec)

    def write(self, data, timeout_sec):
        if self.fd_transport is None:
            raise TransportClosedError()
        return self.fd_transport.write(data, timeout_sec)

    def _qemu_check_stdout(self):
        for line in self.proc.stdout:
            line = str(line)
            _LOG.debug(line)
            if "[QEMU] CPU" in line:
                self._queue.put(ZephyrQemuMakeResult.QEMU_STARTED)
            else:
                line = re.sub("[^a-zA-Z0-9 \n]", "", line)
                pattern = r"recipe for target (\w*) failed"
                if re.search(pattern, line, re.IGNORECASE):
                    self._queue.put(ZephyrQemuMakeResult.MAKE_FAILED)
        self._queue.put(ZephyrQemuMakeResult.EOF)

    def _wait_for_qemu(self):
        threading.Thread(target=self._qemu_check_stdout, daemon=True).start()
        while True:
            try:
                item = self._queue.get(timeout=120)
            except Exception:
                raise TimeoutError("QEMU setup timeout.")

            if item == ZephyrQemuMakeResult.QEMU_STARTED:
                break

            if item in [ZephyrQemuMakeResult.MAKE_FAILED, ZephyrQemuMakeResult.EOF]:
                raise RuntimeError("QEMU setup failed.")

            raise ValueError(f"{item} not expected.")


class ZephyrDebugger(debugger.GdbDebugger):
    """A Zephyr debugger implementation."""

    def __init__(self, west_cmd, build_dir, elf_path, zephyr_base):
        super(ZephyrDebugger, self).__init__()
        self._west_cmd = shlex.split(west_cmd)
        self._build_dir = build_dir
        self._elf_path = elf_path
        self._zephyr_base = zephyr_base

    def popen_kwargs(self):
        env = dict(os.environ)
        env["ZEPHYR_BASE"] = self._zephyr_base

        args = dict(
            args=self._west_cmd
            + [
                "debug",
                "--skip-rebuild",
                "--build-dir",
                self._build_dir,
                "--elf-file",
                self._elf_path,
            ],
            env=env,
        )
        return args
