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

import atexit
import collections
import collections.abc
import enum
import fcntl
import json
import logging
import os
import os.path
import pathlib
import queue
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
from typing import Union
import usb
import psutil
import stat

import serial
import serial.tools.list_ports
import yaml

from tvm.micro.project_api import server


_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR / "build"


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


BOARDS = API_SERVER_DIR / "boards.json"

CMAKELIST_FILENAME = "CMakeLists.txt"

# Used to check Zephyr version installed on the host.
# We only check two levels of the version.
ZEPHYR_VERSION = 2.7

WEST_CMD = default = sys.executable + " -m west" if sys.executable else None

ZEPHYR_BASE = os.getenv("ZEPHYR_BASE")

# Data structure to hold the information microtvm_api_server.py needs
# to communicate with each of these boards.
try:
    with open(BOARDS) as boards:
        BOARD_PROPERTIES = json.load(boards)
except FileNotFoundError:
    raise FileNotFoundError(f"Board file {{{BOARDS}}} does not exist.")


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


CACHE_ENTRY_RE = re.compile(r"(?P<name>[^:]+):(?P<type>[^=]+)=(?P<value>.*)")


CMAKE_BOOL_MAP = dict(
    [(k, True) for k in ("1", "ON", "YES", "TRUE", "Y")]
    + [(k, False) for k in ("0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "")]
)


class CMakeCache(collections.abc.Mapping):
    def __init__(self, path):
        self._path = path
        self._dict = None

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key):
        if self._dict is None:
            self._dict = self._read_cmake_cache()

        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def _read_cmake_cache(self):
        """Read a CMakeCache.txt-like file and return a dictionary of values."""
        entries = collections.OrderedDict()
        with open(self._path, encoding="utf-8") as f:
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


CMAKE_CACHE = CMakeCache(BUILD_DIR / "CMakeCache.txt")


class BoardError(Exception):
    """Raised when an attached board cannot be opened (i.e. missing /dev nodes, etc)."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the board= given to ZephyrCompiler."""


def _get_flash_runner():
    flash_runner = CMAKE_CACHE.get("ZEPHYR_BOARD_FLASH_RUNNER")
    if flash_runner is not None:
        return flash_runner

    with open(CMAKE_CACHE["ZEPHYR_RUNNERS_YAML"]) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc["flash-runner"]


def _find_board_from_cmake_file(cmake_file: Union[str, pathlib.Path]) -> str:
    """Find Zephyr board from generated CMakeLists.txt"""
    zephyr_board = None
    with open(cmake_file) as cmake_f:
        for line in cmake_f:
            if line.startswith("set(BOARD"):
                zephyr_board = line.strip("\n").strip("\r").strip(")").split(" ")[1]
                break

    if not zephyr_board:
        raise RuntimeError(f"No Zephyr board set in the {cmake_file}.")
    return zephyr_board


def _find_platform_from_cmake_file(cmake_file: Union[str, pathlib.Path]) -> str:
    emu_platform = None
    with open(API_SERVER_DIR / CMAKELIST_FILENAME) as cmake_f:
        for line in cmake_f:
            set_platform = re.match("set\(EMU_PLATFORM (.*)\)", line)
            if set_platform:
                emu_platform = set_platform.group(1)
                break
    return emu_platform


def _get_device_args(options):
    flash_runner = _get_flash_runner()

    if flash_runner == "nrfjprog":
        return _get_nrf_device_args(options)

    if flash_runner == "openocd":
        return _get_openocd_device_args(options)

    raise BoardError(
        f"Don't know how to find serial terminal for board {_find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)} with flash "
        f"runner {flash_runner}"
    )


def _get_board_mem_size_bytes(options):
    board_file_path = (
        pathlib.Path(get_zephyr_base(options))
        / "boards"
        / "arm"
        / options["zephyr_board"]
        / (options["zephyr_board"] + ".yaml")
    )
    try:
        with open(board_file_path) as f:
            board_data = yaml.load(f, Loader=yaml.FullLoader)
            return int(board_data["ram"]) * 1024
    except:
        _LOG.warning("Board memory information is not available.")
    return None


DEFAULT_HEAP_SIZE_BYTES = 216 * 1024


def _get_recommended_heap_size_bytes(options):
    prop = BOARD_PROPERTIES[options["zephyr_board"]]
    if "recommended_heap_size_bytes" in prop:
        return prop["recommended_heap_size_bytes"]
    return DEFAULT_HEAP_SIZE_BYTES


def generic_find_serial_port(serial_number=None):
    """Find a USB serial port based on its serial number or its VID:PID.

    This method finds a USB serial port device path based on the port's serial number (if given) or
    based on the board's idVendor and idProduct ids.

    Parameters
    ----------
    serial_number : str
        The serial number associated to the USB serial port which the board is attached to. This is
        the same number as shown by 'lsusb -v' in the iSerial field.

    Returns
    -------
    Path to the USB serial port device, for example /dev/ttyACM1.
    """
    if serial_number:
        regex = serial_number
    else:
        prop = BOARD_PROPERTIES[_find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)]
        device_id = ":".join([prop["vid_hex"], prop["pid_hex"]])
        regex = device_id

    serial_ports = list(serial.tools.list_ports.grep(regex))

    if len(serial_ports) == 0:
        raise Exception(f"No serial port found for board {prop['board']}!")

    if len(serial_ports) != 1:
        ports_lst = ""
        for port in serial_ports:
            ports_lst += f"Serial port: {port.device}, serial number: {port.serial_number}\n"

        raise Exception("Expected 1 serial port, found multiple ports:\n {ports_lst}")

    return serial_ports[0].device


def _get_openocd_device_args(options):
    serial_number = options.get("openocd_serial")
    return ["--serial", generic_find_serial_port(serial_number)]


def _get_nrf_device_args(options):
    nrfjprog_args = ["nrfjprog", "--ids"]
    nrfjprog_ids = subprocess.check_output(nrfjprog_args, encoding="utf-8")
    if not nrfjprog_ids.strip("\n"):
        raise BoardAutodetectFailed(f'No attached boards recognized by {" ".join(nrfjprog_args)}')

    boards = nrfjprog_ids.split("\n")[:-1]
    if len(boards) > 1:
        if options["nrfjprog_snr"] is None:
            raise BoardError(
                "Multiple boards connected; specify one with nrfjprog_snr=: " f'{", ".join(boards)}'
            )

        if str(options["nrfjprog_snr"]) not in boards:
            raise BoardError(
                f"nrfjprog_snr ({options['nrfjprog_snr']}) not found in {nrfjprog_args}: {boards}"
            )

        return ["--snr", options["nrfjprog_snr"]]

    if not boards:
        return []

    return ["--snr", boards[0]]


PROJECT_TYPES = []
if IS_TEMPLATE:
    for d in (API_SERVER_DIR / "src").iterdir():
        if d.is_dir():
            PROJECT_TYPES.append(d.name)


PROJECT_OPTIONS = [
    server.ProjectOption(
        "extra_files_tar",
        optional=["generate_project"],
        type="str",
        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
    ),
    server.ProjectOption(
        "gdbserver_port",
        help=("If given, port number to use when running the local gdbserver."),
        optional=["open_transport"],
        type="int",
    ),
    server.ProjectOption(
        "nrfjprog_snr",
        optional=["open_transport"],
        type="int",
        help=("When used with nRF targets, serial # of the attached board to use, from nrfjprog."),
    ),
    server.ProjectOption(
        "openocd_serial",
        optional=["open_transport"],
        type="int",
        help=("When used with OpenOCD targets, serial # of the attached board to use."),
    ),
    server.ProjectOption(
        "project_type",
        choices=tuple(PROJECT_TYPES),
        required=["generate_project"],
        type="str",
        help="Type of project to generate.",
    ),
    server.ProjectOption(
        "verbose",
        optional=["generate_project"],
        type="bool",
        help="Run build with verbose output.",
    ),
    server.ProjectOption(
        "west_cmd",
        optional=["generate_project"],
        default=WEST_CMD,
        type="str",
        help=(
            "Path to the west tool. If given, supersedes both the zephyr_base "
            "option and ZEPHYR_BASE environment variable."
        ),
    ),
    server.ProjectOption(
        "zephyr_base",
        required=(["generate_project", "open_transport"] if not ZEPHYR_BASE else None),
        optional=(["generate_project", "open_transport"] if ZEPHYR_BASE else ["build"]),
        default=ZEPHYR_BASE,
        type="str",
        help="Path to the zephyr base directory.",
    ),
    server.ProjectOption(
        "zephyr_board",
        required=["generate_project"],
        choices=list(BOARD_PROPERTIES),
        type="str",
        help="Name of the Zephyr board to build for.",
    ),
    server.ProjectOption(
        "config_main_stack_size",
        optional=["generate_project"],
        type="int",
        help="Sets CONFIG_MAIN_STACK_SIZE for Zephyr board.",
    ),
    server.ProjectOption(
        "warning_as_error",
        optional=["generate_project"],
        type="bool",
        help="Treat warnings as errors and raise an Exception.",
    ),
    server.ProjectOption(
        "compile_definitions",
        optional=["generate_project"],
        type="str",
        help="Extra definitions added project compile.",
    ),
    server.ProjectOption(
        "cmsis_path",
        optional=["generate_project"],
        type="str",
        help="Path to the CMSIS directory.",
    ),
    server.ProjectOption(
        "arm_fvp_path",
        optional=["generate_project", "open_transport"],
        type="str",
        help="Path to the FVP binary to invoke.",
    ),
    server.ProjectOption(
        "use_fvp",
        optional=["generate_project"],
        type="bool",
        help="Run on the FVP emulator instead of hardware.",
    ),
    server.ProjectOption(
        "heap_size_bytes",
        optional=["generate_project"],
        type="int",
        help="Sets the value for HEAP_SIZE_BYTES passed to K_HEAP_DEFINE() to service TVM memory allocation requests.",
    ),
]


def get_zephyr_base(options: dict):
    """Returns Zephyr base path"""
    zephyr_base = options.get("zephyr_base", ZEPHYR_BASE)
    assert zephyr_base, "'zephyr_base' option not passed and not found by default!"
    return zephyr_base


def get_cmsis_path(options: dict) -> pathlib.Path:
    """Returns CMSIS dependency path"""
    cmsis_path = options.get("cmsis_path")
    assert cmsis_path, "'cmsis_path' option not passed!"
    return pathlib.Path(cmsis_path)


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="zephyr",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
            project_options=PROJECT_OPTIONS,
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "Makefile", "src")

    # Maps extra line added to prj.conf to a tuple or list of zephyr_board for which it is needed.
    EXTRA_PRJ_CONF_DIRECTIVES = {
        "CONFIG_TIMER_RANDOM_GENERATOR=y": (
            "qemu_x86",
            "qemu_riscv32",
            "qemu_cortex_r5",
            "qemu_riscv64",
        ),
        "CONFIG_ENTROPY_GENERATOR=y": (
            "mps2_an521",
            "nrf5340dk_nrf5340_cpuapp",
            "nucleo_f746zg",
            "nucleo_l4r5zi",
            "stm32f746g_disco",
        ),
    }

    def _create_prj_conf(self, project_dir, options):
        zephyr_board = options["zephyr_board"]
        with open(project_dir / "prj.conf", "w") as f:
            f.write(
                "# For UART used from main().\n"
                "CONFIG_RING_BUFFER=y\n"
                "CONFIG_UART_CONSOLE=n\n"
                "CONFIG_UART_INTERRUPT_DRIVEN=y\n"
                "\n"
            )
            f.write("# For TVMPlatformAbort().\n" "CONFIG_REBOOT=y\n" "\n")

            if options["project_type"] == "host_driven":
                f.write(
                    "CONFIG_TIMING_FUNCTIONS=y\n"
                    "# For RPC server C++ bindings.\n"
                    "CONFIG_CPLUSPLUS=y\n"
                    "CONFIG_LIB_CPLUSPLUS=y\n"
                    "\n"
                )

            f.write("# For math routines\n" "CONFIG_NEWLIB_LIBC=y\n" "\n")

            if self._has_fpu(zephyr_board):
                f.write("# For models with floating point.\n" "CONFIG_FPU=y\n" "\n")

            # Set main stack size, if needed.
            if options.get("config_main_stack_size") is not None:
                f.write(f"CONFIG_MAIN_STACK_SIZE={options['config_main_stack_size']}\n")

            f.write("# For random number generation.\n" "CONFIG_TEST_RANDOM_GENERATOR=y\n")

            f.write("\n# Extra prj.conf directives\n")
            for line, board_list in self.EXTRA_PRJ_CONF_DIRECTIVES.items():
                if zephyr_board in board_list:
                    f.write(f"{line}\n")

            # TODO(mehrdadh): due to https://github.com/apache/tvm/issues/12721
            if zephyr_board not in ["qemu_riscv64"]:
                f.write("# For setting -O2 in compiler.\n" "CONFIG_SPEED_OPTIMIZATIONS=y\n")

            f.write("\n")

    API_SERVER_CRT_LIBS_TOKEN = "<API_SERVER_CRT_LIBS>"
    CMAKE_ARGS_TOKEN = "<CMAKE_ARGS>"
    QEMU_PIPE_TOKEN = "<QEMU_PIPE>"
    CMSIS_PATH_TOKEN = "<CMSIS_PATH>"

    CRT_LIBS_BY_PROJECT_TYPE = {
        "host_driven": "microtvm_rpc_server microtvm_rpc_common aot_executor_module aot_executor common",
        "aot_standalone_demo": "memory microtvm_rpc_common common",
    }

    def _get_platform_version(self, zephyr_base: str) -> float:
        with open(pathlib.Path(zephyr_base) / "VERSION", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace(" ", "").replace("\n", "").replace("\r", "")
                if "VERSION_MAJOR" in line:
                    version_major = line.split("=")[1]
                if "VERSION_MINOR" in line:
                    version_minor = line.split("=")[1]

        return float(f"{version_major}.{version_minor}")

    def _cmsis_required(self, project_path: Union[str, pathlib.Path]) -> bool:
        """Check if CMSIS dependency is required."""
        project_path = pathlib.Path(project_path)
        for path in (project_path / "codegen" / "host" / "src").iterdir():
            if path.is_file():
                with open(path, "r") as lib_f:
                    lib_content = lib_f.read()
                if any(
                    header in lib_content
                    for header in [
                        "<arm_nnsupportfunctions.h>",
                        "arm_nn_types.h",
                        "arm_nnfunctions.h",
                    ]
                ):
                    return True
        return False

    def _generate_cmake_args(self, mlf_extracted_path, options) -> str:
        cmake_args = "\n# cmake args\n"
        if options.get("verbose"):
            cmake_args += "set(CMAKE_VERBOSE_MAKEFILE TRUE)\n"

        if options.get("zephyr_base"):
            cmake_args += f"set(ZEPHYR_BASE {options['zephyr_base']})\n"

        if options.get("west_cmd"):
            cmake_args += f"set(WEST {options['west_cmd']})\n"

        if self._is_qemu(options["zephyr_board"], options.get("use_fvp")):
            # Some boards support more than one emulator, so ensure QEMU is set.
            cmake_args += f"set(EMU_PLATFORM qemu)\n"

        if self._is_fvp(options["zephyr_board"], options.get("use_fvp")):
            cmake_args += "set(EMU_PLATFORM armfvp)\n"
            cmake_args += "set(ARMFVP_FLAGS -I)\n"

        cmake_args += f"set(BOARD {options['zephyr_board']})\n"

        enable_cmsis = self._cmsis_required(mlf_extracted_path)
        if enable_cmsis:
            assert os.environ.get("CMSIS_PATH"), "CMSIS_PATH is not defined."
        cmake_args += f"set(ENABLE_CMSIS {str(enable_cmsis).upper()})\n"

        return cmake_args

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        zephyr_board = options["zephyr_board"]

        # Check Zephyr version
        version = self._get_platform_version(get_zephyr_base(options))
        if version != ZEPHYR_VERSION:
            message = f"Zephyr version found is not supported: found {version}, expected {ZEPHYR_VERSION}."
            if options.get("warning_as_error") is not None and options["warning_as_error"]:
                raise server.ServerError(message=message)
            _LOG.warning(message)

        project_dir = pathlib.Path(project_dir)
        # Make project directory.
        project_dir.mkdir()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Copy boards.json file to generated project.
        shutil.copy2(BOARDS, project_dir / BOARDS.name)

        # Copy overlay files
        board_overlay_path = API_SERVER_DIR / "app-overlay" / f"{zephyr_board}.overlay"
        if board_overlay_path.exists():
            shutil.copy2(board_overlay_path, project_dir / f"{zephyr_board}.overlay")

        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_tar_path = project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        shutil.copy2(model_library_format_path, project_model_library_format_tar_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = os.path.splitext(project_model_library_format_tar_path)[0]
        with tarfile.TarFile(project_model_library_format_tar_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        if self._is_qemu(zephyr_board, options.get("use_fvp")):
            shutil.copytree(API_SERVER_DIR / "qemu-hack", project_dir / "qemu-hack")
        elif self._is_fvp(zephyr_board, options.get("use_fvp")):
            shutil.copytree(API_SERVER_DIR / "fvp-hack", project_dir / "fvp-hack")

        # Populate CRT.
        crt_path = project_dir / "crt"
        crt_path.mkdir()
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Populate Makefile.
        with open(project_dir / CMAKELIST_FILENAME, "w") as cmake_f:
            with open(API_SERVER_DIR / f"{CMAKELIST_FILENAME}.template", "r") as cmake_template_f:
                for line in cmake_template_f:
                    if self.API_SERVER_CRT_LIBS_TOKEN in line:
                        crt_libs = self.CRT_LIBS_BY_PROJECT_TYPE[options["project_type"]]
                        line = line.replace("<API_SERVER_CRT_LIBS>", crt_libs)

                    if self.CMAKE_ARGS_TOKEN in line:
                        line = self._generate_cmake_args(extract_path, options)

                    if self.QEMU_PIPE_TOKEN in line:
                        self.qemu_pipe_dir = pathlib.Path(tempfile.mkdtemp())
                        line = line.replace(self.QEMU_PIPE_TOKEN, str(self.qemu_pipe_dir / "fifo"))

                    if self.CMSIS_PATH_TOKEN in line and self._cmsis_required(extract_path):
                        line = line.replace(self.CMSIS_PATH_TOKEN, str(os.environ["CMSIS_PATH"]))

                    cmake_f.write(line)

                heap_size = _get_recommended_heap_size_bytes(options)
                if options.get("heap_size_bytes"):
                    board_mem_size = _get_board_mem_size_bytes(options)
                    heap_size = options["heap_size_bytes"]
                    if board_mem_size is not None:
                        assert (
                            heap_size < board_mem_size
                        ), f"Heap size {heap_size} is larger than memory size {board_mem_size} on this board."
                cmake_f.write(
                    f"target_compile_definitions(app PUBLIC -DHEAP_SIZE_BYTES={heap_size})\n"
                )

                if options.get("compile_definitions"):
                    flags = options.get("compile_definitions")
                    for item in flags:
                        cmake_f.write(f"target_compile_definitions(app PUBLIC {item})\n")

                if self._is_fvp(zephyr_board, options.get("use_fvp")):
                    cmake_f.write(f"target_compile_definitions(app PUBLIC -DFVP=1)\n")

        self._create_prj_conf(project_dir, options)

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        )

        # Populate src/
        src_dir = project_dir / "src"
        if options["project_type"] != "host_driven" or self._is_fvp(
            zephyr_board, options.get("use_fvp")
        ):
            shutil.copytree(API_SERVER_DIR / "src" / options["project_type"], src_dir)
        else:
            src_dir.mkdir()
            shutil.copy2(API_SERVER_DIR / "src" / options["project_type"] / "main.c", src_dir)

        # Populate extra_files
        if options.get("extra_files_tar"):
            with tarfile.open(options["extra_files_tar"], mode="r:*") as tf:
                tf.extractall(project_dir)

    def build(self, options):
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        BUILD_DIR.mkdir()

        zephyr_board = _find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
        emu_platform = _find_platform_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)

        env = os.environ
        if self._is_fvp(zephyr_board, emu_platform == "armfvp"):
            env["ARMFVP_BIN_PATH"] = str((API_SERVER_DIR / "fvp-hack").resolve())
            # Note: We need to explicitly modify the file permissions and make it an executable to pass CI tests.
            # [To Do]: Move permission change to Build.groovy.j2
            st = os.stat(env["ARMFVP_BIN_PATH"] + "/FVP_Corstone_SSE-300_Ethos-U55")
            os.chmod(
                env["ARMFVP_BIN_PATH"] + "/FVP_Corstone_SSE-300_Ethos-U55",
                st.st_mode | stat.S_IEXEC,
            )

        check_call(["cmake", "-GNinja", ".."], cwd=BUILD_DIR, env=env)

        args = ["ninja"]
        if options.get("verbose"):
            args.append("-v")
        check_call(args, cwd=BUILD_DIR, env=env)

    # A list of all zephyr_board values which are known to launch using QEMU. Many platforms which
    # launch through QEMU by default include "qemu" in their name. However, not all do. This list
    # includes those tested platforms which do not include qemu.
    _KNOWN_QEMU_ZEPHYR_BOARDS = ["mps2_an521", "mps3_an547"]

    # A list of all zephyr_board values which are known to launch using ARM FVP (this script configures
    # Zephyr to use that launch method).
    _KNOWN_FVP_ZEPHYR_BOARDS = ["mps3_an547"]

    @classmethod
    def _is_fvp(cls, board, use_fvp):
        if use_fvp:
            assert (
                board in cls._KNOWN_FVP_ZEPHYR_BOARDS
            ), "FVP can't be used to emulate this board on Zephyr"
            return True
        return False

    @classmethod
    def _is_qemu(cls, board, use_fvp=False):
        return "qemu" in board or (
            board in cls._KNOWN_QEMU_ZEPHYR_BOARDS and not cls._is_fvp(board, use_fvp)
        )

    @classmethod
    def _has_fpu(cls, zephyr_board):
        fpu_boards = [name for name, board in BOARD_PROPERTIES.items() if board["fpu"]]
        return zephyr_board in fpu_boards

    def flash(self, options):
        if _find_platform_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME):
            return  # NOTE: qemu requires no flash step--it is launched from open_transport.

        # The nRF5340DK requires an additional `nrfjprog --recover` before each flash cycle.
        # This is because readback protection is enabled by default when this device is flashed.
        # Otherwise, flashing may fail with an error such as the following:
        #  ERROR: The operation attempted is unavailable due to readback protection in
        #  ERROR: your device. Please use --recover to unlock the device.
        zephyr_board = _find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
        if zephyr_board.startswith("nrf5340dk") and _get_flash_runner() == "nrfjprog":
            recover_args = ["nrfjprog", "--recover"]
            recover_args.extend(_get_nrf_device_args(options))
            check_call(recover_args, cwd=API_SERVER_DIR / "build")

        check_call(["ninja", "flash"], cwd=API_SERVER_DIR / "build")

    def open_transport(self, options):
        zephyr_board = _find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
        emu_platform = _find_platform_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
        if self._is_fvp(zephyr_board, emu_platform == "armfvp"):
            transport = ZephyrFvpTransport(options)
        elif self._is_qemu(zephyr_board):
            transport = ZephyrQemuTransport(options)
        else:
            transport = ZephyrSerialTransport(options)

        to_return = transport.open()
        self._transport = transport
        atexit.register(lambda: self.close_transport())
        return to_return

    def close_transport(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def read_transport(self, n, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.read(n, timeout_sec)

    def write_transport(self, data, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.write(data, timeout_sec)


def _set_nonblock(fd):
    flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
    new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"


class ZephyrSerialTransport:

    NRF5340_VENDOR_ID = 0x1366

    # NRF5340_DK v1.0.0 uses VCOM2
    # NRF5340_DK v2.0.0 uses VCOM1
    NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID = {0x1055: "VCOM2", 0x1051: "VCOM1"}

    @classmethod
    def _lookup_baud_rate(cls, options):
        # TODO(mehrdadh): remove this hack once dtlib.py is a standalone project
        # https://github.com/zephyrproject-rtos/zephyr/blob/v2.7-branch/scripts/dts/README.txt
        sys.path.insert(
            0,
            os.path.join(
                get_zephyr_base(options), "scripts", "dts", "python-devicetree", "src", "devicetree"
            ),
        )
        try:
            import dtlib  # pylint: disable=import-outside-toplevel
        finally:
            sys.path.pop(0)

        dt_inst = dtlib.DT(BUILD_DIR / "zephyr" / "zephyr.dts")
        uart_baud = (
            dt_inst.get_node("/chosen")
            .props["zephyr,console"]
            .to_path()
            .props["current-speed"]
            .to_num()
        )
        _LOG.debug("zephyr transport: found UART baudrate from devicetree: %d", uart_baud)

        return uart_baud

    @classmethod
    def _find_nrf_serial_port(cls, options):
        com_ports = subprocess.check_output(
            ["nrfjprog", "--com"] + _get_device_args(options), encoding="utf-8"
        )
        ports_by_vcom = {}
        for line in com_ports.split("\n")[:-1]:
            parts = line.split()
            ports_by_vcom[parts[2]] = parts[1]

        nrf_board = usb.core.find(idVendor=cls.NRF5340_VENDOR_ID)

        if nrf_board == None:
            raise Exception("_find_nrf_serial_port: unable to find NRF5340DK")

        if nrf_board.idProduct in cls.NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID:
            vcom_port = cls.NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID[nrf_board.idProduct]
        else:
            raise Exception("_find_nrf_serial_port: unable to find known NRF5340DK product ID")

        return ports_by_vcom[vcom_port]

    @classmethod
    def _find_openocd_serial_port(cls, options):
        serial_number = options.get("openocd_serial")
        return generic_find_serial_port(serial_number)

    @classmethod
    def _find_jlink_serial_port(cls, options):
        return generic_find_serial_port()

    @classmethod
    def _find_stm32cubeprogrammer_serial_port(cls, options):
        return generic_find_serial_port()

    @classmethod
    def _find_serial_port(cls, options):
        flash_runner = _get_flash_runner()

        if flash_runner == "nrfjprog":
            return cls._find_nrf_serial_port(options)

        if flash_runner == "openocd":
            return cls._find_openocd_serial_port(options)

        if flash_runner == "jlink":
            return cls._find_jlink_serial_port(options)

        if flash_runner == "stm32cubeprogrammer":
            return cls._find_stm32cubeprogrammer_serial_port(options)

        raise RuntimeError(f"Don't know how to deduce serial port for flash runner {flash_runner}")

    def __init__(self, options):
        self._options = options
        self._port = None

    def open(self):
        port_path = self._find_serial_port(self._options)
        self._port = serial.Serial(port_path, baudrate=self._lookup_baud_rate(self._options))
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=5.0,
            session_established_timeout_sec=5.0,
        )

    def close(self):
        self._port.close()
        self._port = None

    def read(self, n, timeout_sec):
        self._port.timeout = timeout_sec
        to_return = self._port.read(n)
        if not to_return:
            raise server.IoTimeoutError()

        return to_return

    def write(self, data, timeout_sec):
        self._port.write_timeout = timeout_sec
        bytes_written = 0
        while bytes_written < len(data):
            n = self._port.write(data)
            data = data[n:]
            bytes_written += n


class ZephyrQemuMakeResult(enum.Enum):
    QEMU_STARTED = "qemu_started"
    MAKE_FAILED = "make_failed"
    EOF = "eof"


class ZephyrQemuTransport:
    """The user-facing Zephyr QEMU transport class."""

    def __init__(self, options):
        self.options = options
        self.proc = None
        self.pipe_dir = None
        self.read_fd = None
        self.write_fd = None
        self._queue = queue.Queue()

    def open(self):
        with open(BUILD_DIR / "CMakeCache.txt", "r") as cmake_cache_f:
            for line in cmake_cache_f:
                if "QEMU_PIPE:" in line:
                    self.pipe = pathlib.Path(line[line.find("=") + 1 :])
                    break
        self.pipe_dir = self.pipe.parents[0]
        self.write_pipe = self.pipe_dir / "fifo.in"
        self.read_pipe = self.pipe_dir / "fifo.out"
        os.mkfifo(self.write_pipe)
        os.mkfifo(self.read_pipe)

        env = None
        if self.options.get("gdbserver_port"):
            env = os.environ.copy()
            env["TVM_QEMU_GDBSERVER_PORT"] = self.options["gdbserver_port"]

        self.proc = subprocess.Popen(
            ["ninja", "run"],
            cwd=BUILD_DIR,
            env=env,
            stdout=subprocess.PIPE,
        )
        self._wait_for_qemu()

        # NOTE: although each pipe is unidirectional, open both as RDWR to work around a select
        # limitation on linux. Without this, non-blocking I/O can't use timeouts because named
        # FIFO are always considered ready to read when no one has opened them for writing.
        self.read_fd = os.open(self.read_pipe, os.O_RDWR | os.O_NONBLOCK)
        self.write_fd = os.open(self.write_pipe, os.O_RDWR | os.O_NONBLOCK)
        _set_nonblock(self.read_fd)
        _set_nonblock(self.write_fd)

        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=10.0,
            session_established_timeout_sec=10.0,
        )

    def close(self):
        did_write = False
        if self.write_fd is not None:
            try:
                server.write_with_timeout(
                    self.write_fd, b"\x01x", 1.0
                )  # Use a short timeout since we will kill the process
                did_write = True
            except server.IoTimeoutError:
                pass
            os.close(self.write_fd)
            self.write_fd = None

        if self.proc:
            if not did_write:
                self.proc.terminate()
            try:
                self.proc.wait(5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        if self.read_fd:
            os.close(self.read_fd)
            self.read_fd = None

        if self.pipe_dir is not None:
            shutil.rmtree(self.pipe_dir)
            self.pipe_dir = None

    def read(self, n, timeout_sec):
        return server.read_with_timeout(self.read_fd, n, timeout_sec)

    def write(self, data, timeout_sec):
        to_write = bytearray()
        escape_pos = []
        for i, b in enumerate(data):
            if b == 0x01:
                to_write.append(b)
                escape_pos.append(i)
            to_write.append(b)

        while to_write:
            num_written = server.write_with_timeout(self.write_fd, to_write, timeout_sec)
            to_write = to_write[num_written:]

    def _qemu_check_stdout(self):
        for line in self.proc.stdout:
            line = str(line)
            _LOG.info("%s", line)
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


class ZephyrFvpMakeResult(enum.Enum):
    FVP_STARTED = "fvp_started"
    MICROTVM_API_SERVER_INIT = "fvp_initialized"
    MAKE_FAILED = "make_failed"
    EOF = "eof"


class BlockingStream:
    """Reimplementation of Stream class from Iris with blocking semantics."""

    def __init__(self):
        self.q = queue.Queue()
        self.unread = None

    def read(self, n=-1, timeout_sec=None):
        assert (
            n != -1
        ), "expect firmware to open stdin using raw mode, and therefore expect sized read requests"

        data = b""
        if self.unread:
            data = data + self.unread
            self.unread = None

        while len(data) < n:
            try:
                # When there is some data to return, fetch as much as possible, then return what we can.
                # When there is no data yet to return, block.
                data += self.q.get(block=not len(data), timeout=timeout_sec)
            except queue.Empty:
                break

        if len(data) > n:
            self.unread = data[n:]
            data = data[:n]

        return data

    readline = read

    def write(self, data):
        self.q.put(data)


class ZephyrFvpTransport:
    """A transport class that communicates with the ARM FVP via Iris server."""

    def __init__(self, options):
        self.options = options
        self.proc = None
        self._queue = queue.Queue()
        self._import_iris()

    def _import_iris(self):
        assert "arm_fvp_path" in self.options, "arm_fvp_path is not defined."
        # Location as seen in the FVP_Corstone_SSE-300_11.15_24 tar.
        iris_lib_path = (
            pathlib.Path(self.options["arm_fvp_path"]).parent.parent.parent
            / "Iris"
            / "Python"
            / "iris"
        )

        sys.path.insert(0, str(iris_lib_path.parent))
        try:
            import iris.NetworkModelInitializer
        finally:
            sys.path.pop(0)

        self._iris_lib = iris

        def _convertStringToU64Array(strValue):
            numBytes = len(strValue)
            if numBytes == 0:
                return []

            numU64 = (numBytes + 7) // 8
            # Extend the string ending with '\0', so that the string length is multiple of 8.
            # E.g. 'hello' is extended to: 'hello'+\0\0\0
            strExt = strValue.ljust(8 * numU64, b"\0")
            # Convert the string to a list of uint64_t in little endian
            return struct.unpack("<{}Q".format(numU64), strExt)

        iris.iris.convertStringToU64Array = _convertStringToU64Array

    def open(self):
        args = ["ninja"]
        if self.options.get("verbose"):
            args.append("-v")
        args.append("run")
        env = dict(os.environ)
        env["ARMFVP_BIN_PATH"] = str(API_SERVER_DIR / "fvp-hack")
        self.proc = subprocess.Popen(
            args,
            cwd=BUILD_DIR,
            env=env,
            stdout=subprocess.PIPE,
        )
        threading.Thread(target=self._fvp_check_stdout, daemon=True).start()

        self.iris_port = self._wait_for_fvp()
        _LOG.info("IRIS started on port %d", self.iris_port)
        NetworkModelInitializer = self._iris_lib.NetworkModelInitializer.NetworkModelInitializer
        self._model_init = NetworkModelInitializer(
            host="localhost", port=self.iris_port, timeout_in_ms=1000
        )
        self._model = self._model_init.start()
        self._target = self._model.get_target("component.FVP_MPS3_Corstone_SSE_300.cpu0")

        self._target.handle_semihost_io()
        self._target._stdout = BlockingStream()
        self._target._stdin = BlockingStream()
        self._model.run(blocking=False, timeout=100)
        self._wait_for_semihost_init()
        _LOG.info("IRIS semihosting initialized.")

        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=10.0,
            session_established_timeout_sec=10.0,
        )

    def _fvp_check_stdout(self):
        START_MSG = "Iris server started listening to port"
        INIT_MSG = "microTVM Zephyr runtime - running"
        for line in self.proc.stdout:
            line = str(line, "utf-8")
            _LOG.info("%s", line)
            start_msg = re.match(START_MSG + r" ([0-9]+)\n", line)
            init_msg = re.match(INIT_MSG, line)
            if start_msg:
                self._queue.put((ZephyrFvpMakeResult.FVP_STARTED, int(start_msg.group(1))))
            elif init_msg:
                self._queue.put((ZephyrFvpMakeResult.MICROTVM_API_SERVER_INIT, None))
                break
            else:
                line = re.sub("[^a-zA-Z0-9 \n]", "", line)
                pattern = r"recipe for target (\w*) failed"
                if re.search(pattern, line, re.IGNORECASE):
                    self._queue.put((ZephyrFvpMakeResult.MAKE_FAILED, None))

        self._queue.put((ZephyrFvpMakeResult.EOF, None))

    def _wait_for_fvp(self):
        """waiting for the START_MSG to appear on the stdout"""
        while True:
            try:
                item = self._queue.get(timeout=120)
            except Exception:
                raise TimeoutError("FVP setup timeout.")

            if item[0] == ZephyrFvpMakeResult.FVP_STARTED:
                return item[1]

            if item[0] in [ZephyrFvpMakeResult.MAKE_FAILED, ZephyrFvpMakeResult.EOF]:
                raise RuntimeError("FVP setup failed.")

            raise ValueError(f"{item} not expected.")

    def _wait_for_semihost_init(self):
        """waiting for the INIT_MSG to appear on the stdout"""
        while True:
            try:
                item = self._queue.get(timeout=240)
            except Exception:
                raise TimeoutError("semihost init timeout.")

            if item[0] == ZephyrFvpMakeResult.MICROTVM_API_SERVER_INIT:
                return

            raise ValueError(f"{item} not expected.")

    def close(self):
        self._model._shutdown_model()
        self._model.client.disconnect(force=True)
        parent = psutil.Process(self.proc.pid)
        if parent:
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()

    def read(self, n, timeout_sec):
        return self._target.stdout.read(n, timeout_sec)

    def write(self, data, timeout_sec):
        self._target.stdin.write(data)


if __name__ == "__main__":
    server.main(Handler())
