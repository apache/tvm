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

import json
import logging
import os.path
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from string import Template
from packaging import version

from tvm.micro.project_api import server

_LOG = logging.getLogger(__name__)

MODEL_LIBRARY_FORMAT_RELPATH = pathlib.Path("src") / "model" / "model.tar"
API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())
BUILD_DIR = API_SERVER_DIR / "build"
MODEL_LIBRARY_FORMAT_PATH = API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH

IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()

MIN_ARDUINO_CLI_VERSION = version.parse("0.18.0")

BOARDS = API_SERVER_DIR / "boards.json"

ARDUINO_CLI_CMD = shutil.which("arduino-cli")

MAKEFILE_FILENAME = "Makefile"

# Data structure to hold the information microtvm_api_server.py needs
# to communicate with each of these boards.
try:
    with open(BOARDS) as boards:
        BOARD_PROPERTIES = json.load(boards)
except FileNotFoundError:
    raise FileNotFoundError(f"Board file {{{BOARDS}}} does not exist.")


def get_cmsis_path(cmsis_path: pathlib.Path) -> pathlib.Path:
    """Returns CMSIS dependency path"""
    if cmsis_path:
        return pathlib.Path(cmsis_path)
    if os.environ.get("CMSIS_PATH"):
        return pathlib.Path(os.environ.get("CMSIS_PATH"))
    assert False, "'cmsis_path' option not passed!"


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the requested board"""


PROJECT_TYPES = ["example_project", "host_driven"]

PROJECT_OPTIONS = server.default_project_options(
    project_type={"choices": tuple(PROJECT_TYPES)},
    board={"choices": list(BOARD_PROPERTIES), "optional": ["flash", "open_transport"]},
    warning_as_error={"optional": ["build", "flash"]},
) + [
    server.ProjectOption(
        "arduino_cli_cmd",
        required=(["generate_project", "flash", "open_transport"] if not ARDUINO_CLI_CMD else None),
        optional=(
            ["generate_project", "build", "flash", "open_transport"] if ARDUINO_CLI_CMD else None
        ),
        type="str",
        default=ARDUINO_CLI_CMD,
        help="Path to the arduino-cli tool.",
    ),
    server.ProjectOption(
        "port",
        optional=["flash", "open_transport"],
        type="int",
        default=None,
        help="Port to use for connecting to hardware.",
    ),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None
        self._port = None
        self._serial = None
        self._version = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="arduino",
            is_template=IS_TEMPLATE,
            model_library_format_path="" if IS_TEMPLATE else MODEL_LIBRARY_FORMAT_PATH,
            project_options=PROJECT_OPTIONS,
        )

    def _copy_project_files(self, api_server_dir, project_dir, project_type):
        """Copies the files for project_type into project_dir.

        Notes
        -----
        template_dir is NOT a project type, and that directory is never copied
        in this function. template_dir only holds this file and its unit tests,
        so this file is copied separately in generate_project.

        """
        for item in (API_SERVER_DIR / "src" / project_type).iterdir():
            if item.name == "project.ino":
                continue
            dest = project_dir / "src" / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Arduino requires the .ino file have the same filename as its containing folder
        shutil.copy2(
            API_SERVER_DIR / "src" / project_type / "project.ino",
            project_dir / f"{project_dir.stem}.ino",
        )

    CRT_COPY_ITEMS = ("include", "src")

    def _copy_standalone_crt(self, source_dir, standalone_crt_dir):
        output_crt_dir = source_dir / "standalone_crt"
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = output_crt_dir / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    # Example project is the "minimum viable project",
    # and doesn't need a fancy RPC server
    EXAMPLE_PROJECT_UNUSED_COMPONENTS = [
        "include/dmlc",
        "src/support",
        "src/runtime/minrpc",
        "src/runtime/crt/graph_executor",
        "src/runtime/crt/microtvm_rpc_common",
        "src/runtime/crt/microtvm_rpc_server",
        "src/runtime/crt/tab",
    ]

    def _remove_unused_components(self, source_dir, project_type):
        unused_components = []
        if project_type == "example_project":
            unused_components = self.EXAMPLE_PROJECT_UNUSED_COMPONENTS

        for component in unused_components:
            shutil.rmtree(source_dir / "standalone_crt" / component)

    def _disassemble_mlf(self, mlf_tar_path, source_dir):
        with tempfile.TemporaryDirectory() as mlf_unpacking_dir_str:
            mlf_unpacking_dir = pathlib.Path(mlf_unpacking_dir_str)
            with tarfile.open(mlf_tar_path, "r:") as tar:
                tar.extractall(mlf_unpacking_dir)

            model_dir = source_dir / "model"
            model_dir.mkdir()

            # Copy C files from model. The filesnames and quantity
            # depend on the target string, so we just copy all c files
            source_dir = mlf_unpacking_dir / "codegen" / "host" / "src"
            for file in source_dir.rglob("*.c"):
                shutil.copy(file, model_dir)

            # Return metadata.json for use in templating
            with open(os.path.join(mlf_unpacking_dir, "metadata.json")) as f:
                metadata = json.load(f)
        return metadata

    def _template_model_header(self, source_dir, metadata):
        with open(source_dir / "model.h", "r") as f:
            model_h_template = Template(f.read())

        all_module_names = []
        for name in metadata["modules"].keys():
            all_module_names.append(name)

        assert all(
            metadata["modules"][mod_name]["style"] == "full-model" for mod_name in all_module_names
        ), "when generating AOT, expect only full-model Model Library Format"

        workspace_size_bytes = 0
        for mod_name in all_module_names:
            workspace_size_bytes += metadata["modules"][mod_name]["memory"]["functions"]["main"][0][
                "workspace_size_bytes"
            ]
        template_values = {
            "workspace_size_bytes": workspace_size_bytes,
        }

        with open(source_dir / "model.h", "w") as f:
            f.write(model_h_template.substitute(template_values))

    # Arduino ONLY recognizes .ino, .ccp, .c, .h

    CPP_FILE_EXTENSION_SYNONYMS = ("cc", "cxx")

    def _change_cpp_file_extensions(self, source_dir):
        for ext in self.CPP_FILE_EXTENSION_SYNONYMS:
            for filename in source_dir.rglob(f"*.{ext}"):
                filename.rename(filename.with_suffix(".cpp"))

        for filename in source_dir.rglob("*.inc"):
            filename.rename(filename.with_suffix(".h"))

    def _convert_includes(self, project_dir, source_dir):
        """Changes all #include statements in project_dir to be relevant to their
        containing file's location.

        Arduino only supports includes relative to a file's location, so this
        function finds each time we #include a file and changes the path to
        be relative to the file location. Does not do this for standard C
        libraries. Also changes angle brackets syntax to double quotes syntax.

        See Also
        -----
        https://www.arduino.cc/reference/en/language/structure/further-syntax/include/

        """
        for ext in ("c", "h", "cpp"):
            for filename in source_dir.rglob(f"*.{ext}"):
                with filename.open("rb") as src_file:
                    lines = src_file.readlines()
                    with filename.open("wb") as dst_file:
                        for line in lines:
                            line_str = str(line, "utf-8")
                            # Check if line has an include
                            result = re.search(r"#include\s*[<\"]([^>]*)[>\"]", line_str)
                            if not result:
                                dst_file.write(line)
                            else:
                                new_include = self._find_modified_include_path(
                                    project_dir, filename, result.groups()[0]
                                )
                                updated_line = f'#include "{new_include}"\n'
                                dst_file.write(updated_line.encode("utf-8"))

    # Most of the files we used to be able to point to directly are under "src/standalone_crt/include/".
    # Howver, crt_config.h lives under "src/standalone_crt/crt_config/", and more exceptions might
    # be added in the future.
    POSSIBLE_BASE_PATHS = ["src/standalone_crt/include/", "src/standalone_crt/crt_config/"]

    def _find_modified_include_path(self, project_dir, file_path, include_path):
        """Takes a single #include path, and returns the location it should point to.

        Examples
        --------
        >>> _find_modified_include_path(
        ...     "/path/to/project/dir"
        ...     "/path/to/project/dir/src/standalone_crt/src/runtime/crt/common/ndarray.c"
        ...     "tvm/runtime/crt/platform.h"
        ... )
        "../../../../../../src/standalone_crt/include/tvm/runtime/crt/platform.h"

        """
        if include_path.endswith(".inc"):
            include_path = re.sub(r"\.[a-z]+$", ".h", include_path)

        # Change includes referencing .cc and .cxx files to point to the renamed .cpp file
        if include_path.endswith(self.CPP_FILE_EXTENSION_SYNONYMS):
            include_path = re.sub(r"\.[a-z]+$", ".cpp", include_path)

        # If the include already works, don't modify it
        if (file_path.parents[0] / include_path).exists():
            return include_path

        relative_path = file_path.relative_to(project_dir)
        up_dirs_path = "../" * str(relative_path).count("/")

        for base_path in self.POSSIBLE_BASE_PATHS:
            full_potential_path = project_dir / base_path / include_path
            if full_potential_path.exists():
                return up_dirs_path + base_path + include_path

        # If we can't find the file, just leave it untouched
        # It's probably a standard C/C++ header
        return include_path

    CMSIS_INCLUDE_HEADERS = [
        "arm_nn_math_types.h",
        "arm_nn_tables.h",
        "arm_nn_types.h",
        "arm_nnfunctions.h",
        "arm_nnsupportfunctions.h",
    ]

    def _cmsis_required(self, project_path: pathlib.Path) -> bool:
        """Check if CMSIS dependency is required."""
        project_path = pathlib.Path(project_path)
        for path in (project_path / "src" / "model").iterdir():
            if path.is_file():
                # Encoding is for reading C generated code which also includes hex numbers
                with open(path, "r", encoding="ISO-8859-1") as lib_f:
                    lib_content = lib_f.read()
                if any(header in lib_content for header in self.CMSIS_INCLUDE_HEADERS):
                    return True
        return False

    def _copy_cmsis(self, project_path: pathlib.Path, cmsis_path: str):
        """Copy CMSIS header files to project.
        Note: We use this CMSIS package:https://www.arduino.cc/reference/en/libraries/arduino_cmsis-dsp/
        However, the latest release does not include header files that are copied in this function.
        """
        (project_path / "include" / "cmsis").mkdir()
        cmsis_path = get_cmsis_path(cmsis_path)
        for item in self.CMSIS_INCLUDE_HEADERS:
            shutil.copy2(
                cmsis_path / "CMSIS" / "NN" / "Include" / item,
                project_path / "include" / "cmsis" / item,
            )

    def _populate_makefile(
        self,
        makefile_template_path: pathlib.Path,
        makefile_path: pathlib.Path,
        board: str,
        verbose: bool,
        arduino_cli_cmd: str,
        build_extra_flags: str,
    ):
        """Generate Makefile from template."""
        flags = {
            "FQBN": self._get_fqbn(board),
            "VERBOSE_FLAG": "--verbose" if verbose else "",
            "ARUINO_CLI_CMD": self._get_arduino_cli_cmd(arduino_cli_cmd),
            "BOARD": board,
            "BUILD_EXTRA_FLAGS": build_extra_flags,
        }

        with open(makefile_path, "w") as makefile_f:
            with open(makefile_template_path, "r") as makefile_template_f:
                for line in makefile_template_f:
                    SUBST_TOKEN_RE = re.compile(r"<([A-Z_]+)>")
                    outs = []
                    for i, m in enumerate(re.split(SUBST_TOKEN_RE, line)):
                        if i % 2 == 1:
                            m = flags[m]
                        outs.append(m)
                    line = "".join(outs)
                    makefile_f.write(line)

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # List all used project options
        board = options["board"]
        verbose = options.get("verbose")
        project_type = options["project_type"]
        arduino_cli_cmd = options.get("arduino_cli_cmd")
        cmsis_path = options.get("cmsis_path")
        compile_definitions = options.get("compile_definitions")
        extra_files_tar = options.get("extra_files_tar")

        # Reference key directories with pathlib
        project_dir = pathlib.Path(project_dir)
        project_dir.mkdir()
        source_dir = project_dir / "src"
        source_dir.mkdir()

        # Copies files from the template folder to project_dir
        shutil.copy2(API_SERVER_DIR / "microtvm_api_server.py", project_dir)
        shutil.copy2(BOARDS, project_dir / BOARDS.name)
        self._copy_project_files(API_SERVER_DIR, project_dir, project_type)

        # Copy standalone_crt into src folder
        self._copy_standalone_crt(source_dir, standalone_crt_dir)
        self._remove_unused_components(source_dir, project_type)

        # Populate crt-config.h
        crt_config_dir = project_dir / "src" / "standalone_crt" / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        )

        # Unpack the MLF and copy the relevant files
        metadata = self._disassemble_mlf(model_library_format_path, source_dir)
        shutil.copy2(model_library_format_path, project_dir / MODEL_LIBRARY_FORMAT_RELPATH)

        # For AOT, template model.h with metadata to minimize space usage
        if project_type == "example_project":
            self._template_model_header(source_dir, metadata)

        self._change_cpp_file_extensions(source_dir)

        # Recursively change includes
        self._convert_includes(project_dir, source_dir)

        # create include directory
        (project_dir / "include").mkdir()

        # Populate extra_files
        if extra_files_tar:
            with tarfile.open(extra_files_tar, mode="r:*") as tf:
                tf.extractall(project_dir)

        build_extra_flags = '"build.extra_flags='
        if extra_files_tar:
            build_extra_flags += "-I./include "

        if compile_definitions:
            for item in compile_definitions:
                build_extra_flags += f"{item} "

        if self._cmsis_required(project_dir):
            build_extra_flags += f"-I./include/cmsis "
            self._copy_cmsis(project_dir, cmsis_path)

        build_extra_flags += '"'

        # Check if build_extra_flags is empty
        if build_extra_flags == '"build.extra_flags="':
            build_extra_flags = '""'

        # Populate Makefile
        self._populate_makefile(
            API_SERVER_DIR / f"{MAKEFILE_FILENAME}.template",
            project_dir / MAKEFILE_FILENAME,
            board,
            verbose,
            arduino_cli_cmd,
            build_extra_flags,
        )

    def _get_arduino_cli_cmd(self, arduino_cli_cmd: str):
        if not arduino_cli_cmd:
            arduino_cli_cmd = ARDUINO_CLI_CMD
        assert arduino_cli_cmd, "'arduino_cli_cmd' command not passed and not found by default!"
        return arduino_cli_cmd

    def _get_platform_version(self, arduino_cli_path: str) -> float:
        # sample output of this command:
        # 'arduino-cli alpha Version: 0.18.3 Commit: d710b642 Date: 2021-05-14T12:36:58Z\n'
        version_output = subprocess.run(
            [arduino_cli_path, "version"], check=True, stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        str_version = re.search(r"Version: ([\.0-9]*)", version_output).group(1)

        # Using too low a version should raise an error. Note that naively
        # comparing floats will fail here: 0.7 > 0.21, but 0.21 is a higher
        # version (hence we need version.parse)
        return version.parse(str_version)

    # This will only be run for build and upload
    def _check_platform_version(self, cli_command: str, warning_as_error: bool):
        if not self._version:
            self._version = self._get_platform_version(cli_command)

        if self._version < MIN_ARDUINO_CLI_VERSION:
            message = (
                f"Arduino CLI version too old: found {self._version}, "
                f"need at least {str(MIN_ARDUINO_CLI_VERSION)}."
            )
            if warning_as_error is not None and warning_as_error:
                raise server.ServerError(message=message)
            _LOG.warning(message)

    def _get_fqbn(self, board: str):
        o = BOARD_PROPERTIES[board]
        return f"{o['package']}:{o['architecture']}:{o['board']}"

    def build(self, options):
        # List all used project options
        arduino_cli_cmd = options.get("arduino_cli_cmd")
        warning_as_error = options.get("warning_as_error")

        cli_command = self._get_arduino_cli_cmd(arduino_cli_cmd)
        self._check_platform_version(cli_command, warning_as_error)
        compile_cmd = ["make", "build"]
        # Specify project to compile
        subprocess.run(compile_cmd, check=True, cwd=API_SERVER_DIR)

    POSSIBLE_BOARD_LIST_HEADERS = ("Port", "Protocol", "Type", "Board Name", "FQBN", "Core")

    def _parse_connected_boards(self, tabular_str):
        """Parses the tabular output from `arduino-cli board list` into a 2D array

        Examples
        --------
        >>> list(_parse_connected_boards(bytes(
        ...     "Port         Type              Board Name FQBN                          Core               \n"
        ...     "/dev/ttyS4   Serial Port       Unknown                                                     \n"
        ...     "/dev/ttyUSB0 Serial Port (USB) Spresense  SPRESENSE:spresense:spresense SPRESENSE:spresense\n"
        ...     "\n",
        ... "utf-8")))
        [['/dev/ttys4', 'Serial Port', 'Unknown', '', ''], ['/dev/ttyUSB0', 'Serial Port (USB)',
        'Spresense', 'SPRESENSE:spresense:spresense', 'SPRESENSE:spresense']]

        """

        # Which column headers are present depends on the version of arduino-cli
        column_regex = r"\s*|".join(self.POSSIBLE_BOARD_LIST_HEADERS) + r"\s*"
        str_rows = tabular_str.split("\n")
        column_headers = list(re.finditer(column_regex, str_rows[0]))
        assert len(column_headers) > 0

        for str_row in str_rows[1:]:
            if not str_row.strip():
                continue
            device = {}

            for column in column_headers:
                col_name = column.group(0).strip().lower()
                device[col_name] = str_row[column.start() : column.end()].strip()
            yield device

    def _auto_detect_port(self, arduino_cli_cmd: str, board: str) -> str:
        list_cmd = [self._get_arduino_cli_cmd(arduino_cli_cmd), "board", "list"]
        list_cmd_output = subprocess.run(
            list_cmd, check=True, stdout=subprocess.PIPE
        ).stdout.decode("utf-8")

        desired_fqbn = self._get_fqbn(board)
        for device in self._parse_connected_boards(list_cmd_output):
            if device["fqbn"] == desired_fqbn:
                return device["port"]

        # If no compatible boards, raise an error
        raise BoardAutodetectFailed()

    def _get_arduino_port(self, arduino_cli_cmd: str, board: str, port: int):
        if not self._port:
            if port:
                self._port = port
            else:
                self._port = self._auto_detect_port(arduino_cli_cmd, board)

        return self._port

    def _get_board_from_makefile(self, makefile_path: pathlib.Path) -> str:
        """Get Board from generated Makefile."""
        with open(makefile_path) as makefile_f:
            line = makefile_f.readline()
            if "BOARD" in line:
                board = re.sub(r"\s", "", line).split(":=")[1]
                return board
        raise RuntimeError("Board was not found in Makefile: {}".format(makefile_path))

    FLASH_TIMEOUT_SEC = 60
    FLASH_MAX_RETRIES = 5

    def flash(self, options):
        # List all used project options
        arduino_cli_cmd = options.get("arduino_cli_cmd")
        warning_as_error = options.get("warning_as_error")
        port = options.get("port")
        board = options.get("board")
        if not board:
            board = self._get_board_from_makefile(API_SERVER_DIR / MAKEFILE_FILENAME)

        cli_command = self._get_arduino_cli_cmd(arduino_cli_cmd)
        self._check_platform_version(cli_command, warning_as_error)
        port = self._get_arduino_port(cli_command, board, port)

        upload_cmd = ["make", "flash", f"PORT={port}"]
        for _ in range(self.FLASH_MAX_RETRIES):
            try:
                subprocess.run(
                    upload_cmd, check=True, timeout=self.FLASH_TIMEOUT_SEC, cwd=API_SERVER_DIR
                )
                break

            # We only catch timeout errors - a subprocess.CalledProcessError
            # (caused by subprocess.run returning non-zero code) will not
            # be caught.
            except subprocess.TimeoutExpired:
                _LOG.warning(
                    f"Upload attempt to port {port} timed out after {self.FLASH_TIMEOUT_SEC} seconds"
                )

        else:
            raise RuntimeError(
                f"Unable to flash Arduino board after {self.FLASH_MAX_RETRIES} attempts"
            )

    def open_transport(self, options):
        import serial
        import serial.tools.list_ports

        # List all used project options
        arduino_cli_cmd = options.get("arduino_cli_cmd")
        port = options.get("port")
        board = options.get("board")
        if not board:
            board = self._get_board_from_makefile(API_SERVER_DIR / MAKEFILE_FILENAME)

        # Zephyr example doesn't throw an error in this case
        if self._serial is not None:
            return

        port = self._get_arduino_port(arduino_cli_cmd, board, port)

        # It takes a moment for the Arduino code to finish initializing
        # and start communicating over serial
        for _ in range(10):
            if any(serial.tools.list_ports.grep(port)):
                break
            time.sleep(0.5)

        self._serial = serial.Serial(port, baudrate=115200, timeout=10)

        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=5.0,
            session_established_timeout_sec=5.0,
        )

    def close_transport(self):
        if self._serial is None:
            return
        self._serial.close()
        self._serial = None

    def read_transport(self, n, timeout_sec):
        self._serial.timeout = timeout_sec
        if self._serial is None:
            raise server.TransportClosedError()
        return self._serial.read(n)

    def write_transport(self, data, timeout_sec):
        self._serial.write_timeout = timeout_sec
        if self._serial is None:
            raise server.TransportClosedError()
        return self._serial.write(data)


if __name__ == "__main__":
    server.main(Handler())
