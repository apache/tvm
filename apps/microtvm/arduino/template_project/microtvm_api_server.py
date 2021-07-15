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

MODEL_LIBRARY_FORMAT_RELPATH = "src/model/model.tar"

API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())
BUILD_DIR = API_SERVER_DIR / "build"
IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()
MODEL_LIBRARY_FORMAT_PATH = "" if IS_TEMPLATE else API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH
_LOG = logging.getLogger(__name__)

print("Checking if I'm a template:")
print("Yes!" if IS_TEMPLATE else "No!")

class BoardError(Exception):
    """Raised when an attached board cannot be opened (i.e. missing /dev nodes, etc)."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the board= given to ZephyrCompiler."""


PROJECT_OPTIONS = [
    server.ProjectOption("verbose", help="Run build with verbose output"),
    server.ProjectOption("arduino_cmd", help="Path to the arduino-cli tool."),
    server.ProjectOption("arduino_board", help="Name of the Arduino board to build for"),
]

BOARD_PROPERTIES = {
    "spresense": {
        "package": "SPRESENSE",
        "architecture": "spresense",
        "board": "spresense",
    },
    "nano33ble_sense": {
        "package": "arduino",
        "architecture": "mbed_nano",
        "board": "nano33ble",
    }
}


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

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
        "src/runtime/crt/aot_executor",
        "src/runtime/crt/microtvm_rpc_common",
        "src/runtime/crt/microtvm_rpc_server",
        "src/runtime/crt/tab",
    ]
    def _remove_unused_components(self, source_dir):
        for component in self.UNUSED_COMPONENTS:
            shutil.rmtree(source_dir / "standalone_crt" / component)


    GRAPH_JSON_TEMPLATE = 'static const char* graph_json = "{}";\n'
    def _compile_graph_json(self, model_dir, obj):
        graph_json = json.dumps(obj).replace('"', '\\"')
        output = self.GRAPH_JSON_TEMPLATE.format(graph_json)
        graph_json_path = model_dir / "graph_json.c"
        with open(graph_json_path, "w") as out_file:
            out_file.write(output)


    def _disassemble_mlf(self, mlf_tar_path, source_dir):
        mlf_unpacking_dir = tempfile.TemporaryDirectory()
        print(mlf_tar_path)
        with tarfile.open(mlf_tar_path, 'r:') as tar:
            tar.extractall(mlf_unpacking_dir.name)
        print("Unpacked tar")

        # Copy C files
        # TODO are the defaultlib0.c the same?
        model_dir = source_dir / "model"
        model_dir.mkdir()
        for source, dest in [
                ("codegen/host/src/default_lib0.c", "default_lib0.c"),
                ("codegen/host/src/default_lib1.c", "default_lib1.c"),
                ]:
            shutil.copy(
                os.path.join(mlf_unpacking_dir.name, source),
               model_dir / dest
            )

        # Load graph.json, serialize to c format, and extact parameters
        with open(
                os.path.join(mlf_unpacking_dir.name,
                "runtime-config/graph/graph.json")
                ) as f:
            graph_data = json.load(f)
        self._compile_graph_json(model_dir, graph_data)

        mlf_unpacking_dir.cleanup()
        return graph_data

    def _print_c_array(self, l):
        c_arr_str = str(l)
        return "{" + c_arr_str[1:-1] + "}"


    def _print_c_str(self, s):
        return '"{}"'.format(s)


    DL_DATA_TYPE_REFERENCE = {
        "uint8": "{kDLUInt, 8, 0}",
        "uint16": "{kDLUInt, 16, 0}",
        "uint32": "{kDLUInt, 32, 0}",
        "uint64": "{kDLUInt, 64, 0}",
        "int8": "{kDLInt, 8, 0}",
        "int16": "{kDLInt, 16, 0}",
        "int32": "{kDLInt, 32, 0}",
        "int64": "{kDLInt, 64, 0}",
        "float16": "{kDLFloat, 16, 0}",
        "float32": "{kDLFloat, 32, 0}",
        "float64": "{kDLFloat, 64, 0}",
    }
    def _populate_parameters_file(self, graph, source_dir):
        graph_types = graph["attrs"]["dltype"]
        graph_shapes = graph["attrs"]["shape"]
        assert(graph_types[0] == "list_str")
        assert(graph_shapes[0] == "list_shape")

        template_values = {
            "input_data_dimension": len(graph_shapes[1][0]),
            "input_data_shape": self._print_c_array(graph_shapes[1][0]),
            "input_data_type": self.DL_DATA_TYPE_REFERENCE[graph_types[1][0]],
            "output_data_dimension": len(graph_shapes[1][-1]),
            "output_data_shape": self._print_c_array(graph_shapes[1][-1]),
            "output_data_type": self.DL_DATA_TYPE_REFERENCE[graph_types[1][-1]],
            "input_layer_name": self._print_c_str(graph["nodes"][0]["name"]),
        }

        # Apply template values
        with open(source_dir / "parameters.h", 'r') as f:
            template_params = Template(f.read())

        parameters_h = template_params.substitute(template_values)

        with open(source_dir / "parameters.h", "w") as f:
            f.write(parameters_h)


    POSSIBLE_BASE_PATHS = [
        "src/standalone_crt/include/",
        "src/standalone_crt/crt_config/"
    ]
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
        # Copy template folder to project_dir, creating project/ and src/
        # directories in the process. Also copies this file, microtvm_api_server.py,
        # in case TVM needs to call it from the new location
        shutil.copytree(API_SERVER_DIR, project_dir, dirs_exist_ok=True)

        # Reference key directories with pathlib
        project_dir = pathlib.Path(project_dir)
        source_dir = project_dir / "src"

        # Copy standalone_crt into src folder
        self._copy_standalone_crt(source_dir, standalone_crt_dir)
        self._remove_unused_components(source_dir)

        # Unpack the MLF and copy the relevant files
        graph = self._disassemble_mlf(model_library_format_path, source_dir)

        shutil.copy2(model_library_format_path, source_dir / "model")

        # Populate our parameters file
        self._populate_parameters_file(graph, source_dir)

        # Recursively change imports
        self._convert_imports(project_dir, source_dir)


    def _get_fqbn(self, options):
        o = BOARD_PROPERTIES[options['arduino_board']]
        print(o['package'])
        return f"{o['package']}:{o['architecture']}:{o['board']}"


    def build(self, options):
        BUILD_DIR.mkdir()
        print(BUILD_DIR)

        compile_cmd = [
            options['arduino_cmd'], "compile",
            "--fqbn", self._get_fqbn(options),
            "--build-path", BUILD_DIR.resolve()
        ]

        if options.get("verbose"):
            compile_cmd.append("--verbose")

        # Specify project to compile
        compile_cmd.append("./project/")
        print(compile_cmd)
        print(API_SERVER_DIR)
        subprocess.check_call(compile_cmd)


    def flash(self, options):
        raise NotImplementedError


    def open_transport(self, options):
        raise NotImplementedError


    def close_transport(self):
        raise NotImplementedError


    def read_transport(self, n, timeout_sec):
        raise NotImplementedError


    def write_transport(self, data, timeout_sec):
        raise NotImplementedError


if __name__ == "__main__":
    server.main(Handler())
