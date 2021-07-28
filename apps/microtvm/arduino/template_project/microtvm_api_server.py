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
from string import Template
import tempfile
import time

import serial
import serial.tools.list_ports

from tvm.micro.project_api import server

MODEL_LIBRARY_FORMAT_RELPATH = pathlib.Path("src") / "model" / "model.tar"
API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())
BUILD_DIR = API_SERVER_DIR / "build"
MODEL_LIBRARY_FORMAT_PATH = API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH

IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


class InvalidPortException(Exception):
    """Raised when the given port could not be opened"""


class SketchUploadException(Exception):
    """Raised when a sketch cannot be uploaded for an unknown reason."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the requested board"""


BOARD_PROPERTIES = {
    "spresense": {
        "package": "SPRESENSE",
        "architecture": "spresense",
        "board": "spresense",
    },
    "nano33ble": {
        "package": "arduino",
        "architecture": "mbed_nano",
        "board": "nano33ble",
    },
}

PROJECT_OPTIONS = [
    server.ProjectOption(
        "arduino_board",
        choices=list(BOARD_PROPERTIES),
        help="Name of the Arduino board to build for",
    ),
    server.ProjectOption("arduino_cli_cmd", help="Path to the arduino-cli tool."),
    server.ProjectOption("port", help="Port to use for connecting to hardware"),
    server.ProjectOption(
        "verbose", help="True to pass --verbose flag to arduino-cli compile and upload"
    ),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None
        self._port = None
        self._serial = None

    def server_info_query(self):
        return server.ServerInfo(
            platform_name="arduino",
            is_template=IS_TEMPLATE,
            model_library_format_path=MODEL_LIBRARY_FORMAT_PATH,
            project_options=PROJECT_OPTIONS,
        )

    CRT_COPY_ITEMS = ("include", "src")

    def _copy_standalone_crt(self, source_dir, standalone_crt_dir):
        # Copy over the standalone_crt directory
        output_crt_dir = source_dir / "standalone_crt"
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = output_crt_dir / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    UNUSED_COMPONENTS = [
        "include/dmlc",
        "src/support",
        "src/runtime/minrpc",
        "src/runtime/crt/graph_executor",
        "src/runtime/crt/microtvm_rpc_common",
        "src/runtime/crt/microtvm_rpc_server",
        "src/runtime/crt/tab",
    ]

    def _remove_unused_components(self, source_dir):
        for component in self.UNUSED_COMPONENTS:
            shutil.rmtree(source_dir / "standalone_crt" / component)

    def _disassemble_mlf(self, mlf_tar_path, source_dir):
        with tempfile.TemporaryDirectory() as mlf_unpacking_dir:
            with tarfile.open(mlf_tar_path, "r:") as tar:
                tar.extractall(mlf_unpacking_dir)
                tar.extractall("/home/guberti/mlf_out")

            # Copy C files
            model_dir = source_dir / "model"
            model_dir.mkdir()
            for source, dest in [
                ("codegen/host/src/default_lib0.c", "default_lib0.c"),
                ("codegen/host/src/default_lib1.c", "default_lib1.c"),
            ]:
                shutil.copy(os.path.join(mlf_unpacking_dir, source), model_dir / dest)

            # Return metadata.json for use in templating
            with open(os.path.join(mlf_unpacking_dir, "metadata.json")) as f:
                metadata = json.load(f)
        return metadata

    def _template_model_header(self, source_dir, metadata):
        with open(source_dir / "model.h", "r") as f:
            model_h_template = Template(f.read())

        template_values = {
            "workspace_size_bytes": metadata["memory"]["functions"]["main"][0][
                "workspace_size_bytes"
            ],
        }

        with open(source_dir / "model.h", "w") as f:
            f.write(model_h_template.substitute(template_values))

    POSSIBLE_BASE_PATHS = ["src/standalone_crt/include/", "src/standalone_crt/crt_config/"]

    def _find_modified_include_path(self, project_dir, file_path, import_path):

        # If the import already works, don't modify it
        if (file_path.parents[0] / import_path).exists():
            return import_path

        relative_path = file_path.relative_to(project_dir)
        up_dirs_path = "../" * str(relative_path).count("/")

        for base_path in self.POSSIBLE_BASE_PATHS:
            full_potential_path = project_dir / base_path / import_path
            if full_potential_path.exists():
                new_include = up_dirs_path + base_path + import_path
                return new_include

        # If we can't find the file, just leave it untouched
        # It's probably a standard C/C++ header
        return import_path

    # Arduino only supports imports relative to the top-level project,
    # so we need to adjust each import to meet this convention
    def _convert_imports(self, project_dir, source_dir):
        for ext in ("c", "h"):
            for filename in source_dir.rglob(f"*.{ext}"):
                with filename.open() as file:
                    lines = file.readlines()

                for i in range(len(lines)):
                    # Check if line has an include
                    result = re.search(r"#include\s*[<\"]([^>]*)[>\"]", lines[i])
                    if not result:
                        continue
                    new_include = self._find_modified_include_path(
                        project_dir, filename, result.groups()[0]
                    )

                    lines[i] = f'#include "{new_include}"\n'

                with filename.open("w") as file:
                    file.writelines(lines)

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Reference key directories with pathlib
        project_dir = pathlib.Path(project_dir)
        source_dir = project_dir / "src"

        # Copies files from the template folder to project_dir. model.h is copied here,
        # but will also need to be templated later.
        if IS_TEMPLATE:
            shutil.copytree(API_SERVER_DIR / "src", source_dir, dirs_exist_ok=True)
            shutil.copy2(API_SERVER_DIR / "microtvm_api_server.py", project_dir)

            # Arduino requires the .ino file have the same filename as its containing folder
            shutil.copy2(API_SERVER_DIR / "project.ino", project_dir / f"{project_dir.stem}.ino")

        # Copy standalone_crt into src folder
        self._copy_standalone_crt(source_dir, standalone_crt_dir)
        self._remove_unused_components(source_dir)

        # Unpack the MLF and copy the relevant files
        metadata = self._disassemble_mlf(model_library_format_path, source_dir)
        shutil.copy2(model_library_format_path, source_dir / "model")

        # Template model.h with metadata to minimize space usage
        self._template_model_header(source_dir, metadata)

        # Recursively change imports
        self._convert_imports(project_dir, source_dir)

    def _get_fqbn(self, options):
        o = BOARD_PROPERTIES[options["arduino_board"]]
        return f"{o['package']}:{o['architecture']}:{o['board']}"

    def build(self, options):
        BUILD_DIR.mkdir()
        print(BUILD_DIR)

        compile_cmd = [
            options["arduino_cli_cmd"],
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
        output = subprocess.check_call(compile_cmd)
        assert output == 0

    # We run the command `arduino-cli board list`, which produces
    # outputs of the form:
    """
    Port         Type              Board Name FQBN                          Core
    /dev/ttyS4   Serial Port       Unknown
    /dev/ttyUSB0 Serial Port (USB) Spresense  SPRESENSE:spresense:spresense SPRESENSE:spresense
    """

    def _auto_detect_port(self, options):
        list_cmd = [options["arduino_cli_cmd"], "board", "list"]
        list_cmd_output = subprocess.check_output(list_cmd).decode("utf-8")
        # Remove header and new lines at bottom
        port_options = list_cmd_output.split("\n")[1:-2]

        # Select the first compatible board
        fqbn = self._get_fqbn(options)
        for port_option in port_options:
            if fqbn in port_option:
                return port_option.split(" ")[0]

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
        port = self._get_arduino_port(options)

        upload_cmd = [
            options["arduino_cli_cmd"],
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
            compile_cmd.append("--verbose")

        output = subprocess.check_call(upload_cmd)

        if output == 2:
            raise InvalidPortException()
        elif output > 0:
            raise SketchUploadException()

    def open_transport(self, options):
        # Zephyr example doesn't throw an error in this case
        if self._serial is not None:
            return

        port = self._get_arduino_port(options)
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
        return self._serial.write(data, timeout_sec)


if __name__ == "__main__":
    server.main(Handler())
