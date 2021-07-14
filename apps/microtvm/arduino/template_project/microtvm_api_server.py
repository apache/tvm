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


_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR / "build"


MODEL_LIBRARY_FORMAT_RELPATH = "src/model/model.tar"


IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


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
        entries = collections.abc.OrderedDict()
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


'''def _get_flash_runner():
    flash_runner = CMAKE_CACHE.get("ZEPHYR_BOARD_FLASH_RUNNER")
    if flash_runner is not None:
        return flash_runner

    with open(CMAKE_CACHE["ZEPHYR_RUNNERS_YAML"]) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    return doc["flash-runner"]


def _get_device_args(options, cmake_entries):
    flash_runner = _get_flash_runner()

    if flash_runner == "nrfjprog":
        return _get_nrf_device_args(options)

    if flash_runner == "openocd":
        return _get_openocd_device_args(options)

    raise BoardError(
        f"Don't know how to find serial terminal for board {CMAKE_CACHE['BOARD']} with flash "
        f"runner {flash_runner}"
    )'''


# kwargs passed to usb.core.find to find attached boards for the openocd flash runner.
BOARD_USB_FIND_KW = {
    "nucleo_l4r5zi": {"idVendor": 0x0483, "idProduct": 0x374B},
    "nucleo_f746zg": {"idVendor": 0x0483, "idProduct": 0x374B},
    "stm32f746g_disco": {"idVendor": 0x0483, "idProduct": 0x374B},
}


def openocd_serial(options):
    """Find the serial port to use for a board with OpenOCD flash strategy."""
    if "openocd_serial" in options:
        return options["openocd_serial"]

    import usb  # pylint: disable=import-outside-toplevel

    find_kw = BOARD_USB_FIND_KW[CMAKE_CACHE["BOARD"]]
    boards = usb.core.find(find_all=True, **find_kw)
    serials = []
    for b in boards:
        serials.append(b.serial_number)

    if len(serials) == 0:
        raise BoardAutodetectFailed(f"No attached USB devices matching: {find_kw!r}")
    serials.sort()

    autodetected_openocd_serial = serials[0]
    _LOG.debug("zephyr openocd driver: autodetected serial %s", serials[0])

    return autodetected_openocd_serial


def _get_openocd_device_args(options):
    return ["--serial", openocd_serial(options)]


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


PROJECT_OPTIONS = [
    server.ProjectOption(
        "gdbserver_port", help=("If given, port number to use when running the " "local gdbserver")
    ),
    server.ProjectOption(
        "openocd_serial",
        help=("When used with OpenOCD targets, serial # of the " "attached board to use"),
    ),
    server.ProjectOption(
        "nrfjprog_snr",
        help=(
            "When used with nRF targets, serial # of the " "attached board to use, from nrfjprog"
        ),
    ),
    server.ProjectOption("verbose", help="Run build with verbose output"),
    server.ProjectOption(
        "west_cmd",
        help=(
            "Path to the west tool. If given, supersedes both the zephyr_base "
            "option and ZEPHYR_BASE environment variable."
        ),
    ),
    server.ProjectOption("zephyr_base", help="Path to the zephyr base directory."),
    server.ProjectOption("zephyr_board", help="Name of the Zephyr board to build for"),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self):
        return server.ServerInfo(
            platform_name="arduino",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
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

        # Populate our parameters file
        self._populate_parameters_file(graph, source_dir)

        # Recursively change imports
        self._convert_imports(project_dir, source_dir)

    def build(self, options):
        BUILD_DIR.mkdir()

        cmake_args = ["cmake", ".."]
        if options.get("verbose"):
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE")

        if options.get("zephyr_base"):
            cmake_args.append(f"-DZEPHYR_BASE:STRING={options['zephyr_base']}")

        cmake_args.append(f"-DBOARD:STRING={options['zephyr_board']}")

        check_call(cmake_args, cwd=BUILD_DIR)

        args = ["make", "-j2"]
        if options.get("verbose"):
            args.append("VERBOSE=1")
        check_call(args, cwd=BUILD_DIR)

    @classmethod
    def _is_qemu(cls, options):
        return "qemu" in options["zephyr_board"]

    def flash(self, options):
        if self._is_qemu(options):
            return  # NOTE: qemu requires no flash step--it is launched from open_transport.

        zephyr_board = options["zephyr_board"]

        # The nRF5340DK requires an additional `nrfjprog --recover` before each flash cycle.
        # This is because readback protection is enabled by default when this device is flashed.
        # Otherwise, flashing may fail with an error such as the following:
        #  ERROR: The operation attempted is unavailable due to readback protection in
        #  ERROR: your device. Please use --recover to unlock the device.
        if zephyr_board.startswith("nrf5340dk") and _get_flash_runner() == "nrfjprog":
            recover_args = ["nrfjprog", "--recover"]
            recover_args.extend(_get_nrf_device_args(options))
            self._subprocess_env.run(recover_args, cwd=build_dir)

        check_call(["make", "flash"], cwd=API_SERVER_DIR / "build")

    def _open_qemu_transport(self, options):
        zephyr_board = options["zephyr_board"]
        # For Zephyr boards that run emulated by default but don't have the prefix "qemu_" in their
        # board names, a suffix "-qemu" is added by users of µTVM when specifying the board name to
        # inform that the QEMU transporter must be used just like for the boards with the prefix.
        # Zephyr does not recognize the suffix, so we trim it off before passing it.
        if "-qemu" in zephyr_board:
            zephyr_board = zephyr_board.replace("-qemu", "")

        return ZephyrQemuTransport(options)

    def open_transport(self, options):
        if self._is_qemu(options):
            transport = self._open_qemu_transport(options)
        else:
            transport = ZephyrSerialTransport(options)

        to_return = transport.open()
        self._transport = transport
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


if __name__ == "__main__":
    server.main(Handler())
