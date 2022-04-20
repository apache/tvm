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

import collections
import functools
import json
import logging
import os
import os.path
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from string import Template
import re

from packaging import version
import serial.tools.list_ports

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

# Data structure to hold the information microtvm_api_server.py needs
# to communicate with each of these boards.
try:
    with open(BOARDS) as boards:
        BOARD_PROPERTIES = json.load(boards)
except FileNotFoundError:
    raise FileNotFoundError(f"Board file {{{BOARDS}}} does not exist.")


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the requested board"""


PROJECT_TYPES = ["example_project", "host_driven"]

PROJECT_OPTIONS = [
    server.ProjectOption(
        "arduino_board",
        required=["build", "flash", "open_transport"],
        choices=list(BOARD_PROPERTIES),
        type="str",
        help="Name of the Arduino board to build for.",
    ),
    server.ProjectOption(
        "arduino_cli_cmd",
        required=(
            ["generate_project", "build", "flash", "open_transport"]
            if not ARDUINO_CLI_CMD
            else None
        ),
        optional=(
            ["generate_project", "build", "flash", "open_transport"] if ARDUINO_CLI_CMD else None
        ),
        default=ARDUINO_CLI_CMD,
        type="str",
        help="Path to the arduino-cli tool.",
    ),
    server.ProjectOption(
        "port",
        optional=["flash", "open_transport"],
        type="int",
        help="Port to use for connecting to hardware.",
    ),
    server.ProjectOption(
        "project_type",
        required=["generate_project"],
        choices=tuple(PROJECT_TYPES),
        type="str",
        help="Type of project to generate.",
    ),
    server.ProjectOption(
        "verbose",
        optional=["build", "flash"],
        type="bool",
        help="Run arduino-cli compile and upload with verbose output.",
    ),
    server.ProjectOption(
        "warning_as_error",
        optional=["build", "flash"],
        type="bool",
        help="Treat warnings as errors and raise an Exception.",
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
            for file in source_dir.rglob(f"*.c"):
                shutil.copy(file, model_dir)

            # Return metadata.json for use in templating
            with open(os.path.join(mlf_unpacking_dir, "metadata.json")) as f:
                metadata = json.load(f)
        return metadata

    def _template_model_header(self, source_dir, metadata):
        with open(source_dir / "model.h", "r") as f:
            model_h_template = Template(f.read())

        assert (
            metadata["style"] == "full-model"
        ), "when generating AOT, expect only full-model Model Library Format"

        template_values = {
            "workspace_size_bytes": metadata["memory"]["functions"]["main"][0][
                "workspace_size_bytes"
            ],
        }

        with open(source_dir / "model.h", "w") as f:
            f.write(model_h_template.substitute(template_values))

    # Arduino ONLY recognizes .ino, .ccp, .c, .h

    CPP_FILE_EXTENSION_SYNONYMS = ("cc", "cxx")

    def _change_cpp_file_extensions(self, source_dir):
        for ext in self.CPP_FILE_EXTENSION_SYNONYMS:
            for filename in source_dir.rglob(f"*.{ext}"):
                filename.rename(filename.with_suffix(".cpp"))

        for filename in source_dir.rglob(f"*.inc"):
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
                        for i, line in enumerate(lines):
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

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Reference key directories with pathlib
        project_dir = pathlib.Path(project_dir)
        project_dir.mkdir()
        source_dir = project_dir / "src"
        source_dir.mkdir()

        # Copies files from the template folder to project_dir
        shutil.copy2(API_SERVER_DIR / "microtvm_api_server.py", project_dir)
        shutil.copy2(BOARDS, project_dir / BOARDS.name)
        self._copy_project_files(API_SERVER_DIR, project_dir, options["project_type"])

        # Copy standalone_crt into src folder
        self._copy_standalone_crt(source_dir, standalone_crt_dir)
        self._remove_unused_components(source_dir, options["project_type"])

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
        if options["project_type"] == "example_project":
            self._template_model_header(source_dir, metadata)

        self._change_cpp_file_extensions(source_dir)

        # Recursively change includes
        self._convert_includes(project_dir, source_dir)

    def _get_arduino_cli_cmd(self, options: dict):
        arduino_cli_cmd = options.get("arduino_cli_cmd", ARDUINO_CLI_CMD)
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
    def _check_platform_version(self, options):
        if not self._version:
            cli_command = self._get_arduino_cli_cmd(options)
            self._version = self._get_platform_version(cli_command)

        if self._version < MIN_ARDUINO_CLI_VERSION:
            message = (
                f"Arduino CLI version too old: found {self._version}, "
                f"need at least {str(MIN_ARDUINO_CLI_VERSION)}."
            )
            if options.get("warning_as_error") is not None and options["warning_as_error"]:
                raise server.ServerError(message=message)
            _LOG.warning(message)

    def _get_fqbn(self, options):
        o = BOARD_PROPERTIES[options["arduino_board"]]
        return f"{o['package']}:{o['architecture']}:{o['board']}"

    def build(self, options):
        self._check_platform_version(options)
        BUILD_DIR.mkdir()

        compile_cmd = [
            self._get_arduino_cli_cmd(options),
            "compile",
            "./project/",
            "--fqbn",
            self._get_fqbn(options),
            "--build-path",
            BUILD_DIR.resolve(),
        ]

        if options.get("verbose"):
            compile_cmd.append("--verbose")

        # Specify project to compile
        subprocess.run(compile_cmd, check=True)

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

    def _auto_detect_port(self, options):
        list_cmd = [self._get_arduino_cli_cmd(options), "board", "list"]
        list_cmd_output = subprocess.run(
            list_cmd, check=True, stdout=subprocess.PIPE
        ).stdout.decode("utf-8")

        desired_fqbn = self._get_fqbn(options)
        for device in self._parse_connected_boards(list_cmd_output):
            if device["fqbn"] == desired_fqbn:
                return device["port"]

        # If no compatible boards, raise an error
        raise BoardAutodetectFailed()

    def _get_arduino_port(self, options):
        if not self._port:
            if "port" in options and options["port"]:
                self._port = options["port"]
            else:
                self._port = self._auto_detect_port(options)

        return self._port

    def flash(self, options):
        self._check_platform_version(options)
        port = self._get_arduino_port(options)

        upload_cmd = [
            self._get_arduino_cli_cmd(options),
            "upload",
            "./project",
            "--fqbn",
            self._get_fqbn(options),
            "--input-dir",
            BUILD_DIR.resolve(),
            "--port",
            port,
        ]

        if options.get("verbose"):
            upload_cmd.append("--verbose")

        subprocess.run(upload_cmd, check=True)

    def open_transport(self, options):
        # Zephyr example doesn't throw an error in this case
        if self._serial is not None:
            return

        port = self._get_arduino_port(options)

        # It takes a moment for the Arduino code to finish initializing
        # and start communicating over serial
        for attempts in range(10):
            if any(serial.tools.list_ports.grep(port)):
                break
            time.sleep(0.5)

        self._serial = serial.Serial(port, baudrate=115200, timeout=5)

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
        # It's hard to set timeout_sec, so we just throw it away
        # TODO fix this
        if self._serial is None:
            raise server.TransportClosedError()
        return self._serial.read(n)

    def write_transport(self, data, timeout_sec):
        if self._serial is None:
            raise server.TransportClosedError()
        return self._serial.write(data)


if __name__ == "__main__":
    server.main(Handler())
