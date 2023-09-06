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
# pylint: disable=invalid-name, f-string-without-interpolation, consider-using-from-import
"""Tools/compilers/linkers for Hexagon"""

import os
import pathlib
import re
from typing import List, Union
import subprocess
import sys
import tarfile
import io
import numpy

import tvm
import tvm.contrib.cc as cc
from ..._ffi.registry import register_func


# Linking Hexagon shared libraries.
#
#   link_shared(name-of-shared-library, list-of-objects, kw-args)
#
# To use a custom linker, define a function that returns the path to the
# linker, and pass it to 'register_linker':
#
#   def custom_linker_path():
#       return '/path/to/hexagon/linker'
#
#   register_linker(custom_linker_path)
#
# Subsequent calls to 'link_shared' will use the newly registered linker.

HEXAGON_TOOLCHAIN = os.environ.get("HEXAGON_TOOLCHAIN", default="")  # pylint: disable=invalid-name
HEXAGON_SDK_ROOT = os.environ.get("HEXAGON_SDK_ROOT", default="")  # pylint: disable=invalid-name
HEXAGON_SDK_DOCKER_IMAGE = os.environ.get(
    "HEXAGON_SDK_DOCKER_IMAGE", default=""
)  # pylint: disable=invalid-name
HEXAGON_LINK_MAIN = (
    pathlib.Path(HEXAGON_TOOLCHAIN) / "bin" / "hexagon-link"
)  # pylint: disable=invalid-name
HEXAGON_CLANG_PLUS = (
    pathlib.Path(HEXAGON_TOOLCHAIN) / "bin" / "hexagon-clang++"
)  # pylint: disable=invalid-name
HEXAGON_SDK_INCLUDE_DIRS = [  # pylint: disable=invalid-name
    pathlib.Path(HEXAGON_SDK_ROOT) / "incs",
    pathlib.Path(HEXAGON_SDK_ROOT) / "incs" / "stddef",
]

HEXAGON_SIMULATOR_NAME = "simulator"


def register_linker(f):
    """Register a function that will return the path to the Hexagon linker."""
    return register_func("tvm.contrib.hexagon.hexagon_link", f, True)


@register_func("tvm.contrib.hexagon.hexagon_link")
def hexagon_link() -> str:
    """Return path to the Hexagon linker."""
    return str(HEXAGON_LINK_MAIN)


def hexagon_clang_plus() -> str:
    """Return path to the Hexagon clang++."""
    return str(HEXAGON_CLANG_PLUS)


def toolchain_version(toolchain=None) -> List[int]:
    """Return the version of the Hexagon toolchain.

    Parameters
    ----------
    toolchain: str, optional
        Path to the Hexagon toolchain. If not provided, the environment
        variable HEXAGON_TOOLCHAIN is used.

    Returns
    -------
    version: List[int]
        List of numerical components of the version number. E.g. for version
        "8.5.06" it will be [8, 5, 6].
    """

    if toolchain is None:
        toolchain = HEXAGON_TOOLCHAIN
    assert toolchain is not None, "Please specify toolchain, or set HEXAGON_TOOLCHAIN variable"
    result = subprocess.run(
        [f"{toolchain}/bin/hexagon-clang", "-v"], capture_output=True, check=True
    )
    output = result.stderr.decode()
    for line in output.splitlines():
        m = re.match(r".* [Cc]lang version ([0-9\.]+)", line)
        if m:
            assert len(m.groups()) == 1
            return [int(v) for v in m.group(1).split(".")]
    raise RuntimeError("Cannot establish toolchain version")


@register_func("tvm.contrib.hexagon.link_shared")
def link_shared(so_name, objs, extra_args=None):
    """Link shared library on Hexagon using the registered Hexagon linker.

    Parameters
    ----------
    so_name : str
        Name of the shared library file.
    objs : list[str,StringImm]
    extra_args : dict (str->str) or Map<String,String>
        Additional arguments:
            'hex_arch' - Hexagon architecture, e.g. v68
            'verbose'  - Print additional information if the key is present

    Returns
    -------
    ret_val : int
        This function returns 0 at the moment.
    """

    # The list of object files can be passed as built-in Python strings,
    # or as tvm.tir.StringImm's.
    def to_str(s):
        if isinstance(s, tvm.tir.StringImm):
            return s.value
        assert isinstance(s, str), 'argument "' + str(s) + '" should be a string or StrImm'
        return s

    objs = [to_str(s) for s in objs]

    if not extra_args:
        extra_args = {}
    hex_arch = extra_args.get("hex_arch") or "v68"
    linker = tvm.get_global_func("tvm.contrib.hexagon.hexagon_link")()
    if extra_args.get("verbose"):
        print("tvm.contrib.hexagon.link_shared:")
        print("  Using linker:", linker)
        print("  Library name:", so_name)
        print("  Object files:", objs)
        print("  Architecture:", hex_arch)
    if not os.access(linker, os.X_OK):
        message = 'The linker "' + linker + '" does not exist or is not executable.'
        if not os.environ.get("HEXAGON_TOOLCHAIN"):
            message += (
                " The environment variable HEXAGON_TOOLCHAIN is unset. Please export "
                + "HEXAGON_TOOLCHAIN in your environment, so that ${HEXAGON_TOOLCHAIN}/bin/"
                + "hexagon-link exists."
            )
        else:
            message += (
                " Please verify the value of the HEXAGON_LINKER environment variable "
                + '(currently set to "'
                + HEXAGON_TOOLCHAIN
                + '").'
            )
        raise Exception(message)

    libpath = os.path.join(HEXAGON_TOOLCHAIN, "target", "hexagon", "lib", hex_arch, "G0")
    cc.create_shared(
        so_name,
        objs,
        # pylint: disable=bad-whitespace
        options=[
            "-Bdynamic",
            "-shared",
            "-export-dynamic",
            os.path.join(libpath, "pic", "libgcc.so"),
        ],
        cc=linker,
    )
    return 0


def link_shared_macos(so_name, objs, extra_args=None):
    """Link Hexagon shared library using docker container with proper tooling.

    Parameters
    ----------
    so_name : str
        Name of the shared library file.
    objs : list[str,StringImm]
    extra_args : dict (str->str) or Map<String,String>
        Additional arguments:
            'hex_arch' - Hexagon architecture, e.g. v68

    Returns
    -------
    ret_val : int
        This function returns 0 at the moment.
    """

    # The list of object files can be passed as built-in Python strings,
    # or as tvm.tir.StringImm's.
    def to_str(s):
        if isinstance(s, tvm.tir.StringImm):
            return s.value
        assert isinstance(s, str), 'argument "' + str(s) + '" should be a string or StrImm'
        return s

    objs = [to_str(s) for s in objs]

    if not extra_args:
        extra_args = {}
    hex_arch = extra_args.get("hex_arch") or "v68"

    ses = ContainerSession(HEXAGON_SDK_DOCKER_IMAGE)

    hexagon_sdk_tools_path = ses.get_env("HEXAGON_TOOLCHAIN")
    libpath = os.path.join(hexagon_sdk_tools_path, "target", "hexagon", "lib", hex_arch, "G0")
    linker = os.path.join(hexagon_sdk_tools_path, "bin", "hexagon-link")

    # Copy input data to docker container
    docker_objs = [ses.copy_to(obj) for obj in objs]
    docker_so_name = ses.tmp_dir + "/" + os.path.basename(so_name)

    link_cmd = [linker, "-shared", "-fPIC", "-o", docker_so_name]
    link_cmd += docker_objs
    link_cmd += [
        "-Bdynamic",
        "-export-dynamic",
        "-L" + os.path.join(libpath, "pic"),
        "-lgcc",
    ]
    ses.exec(link_cmd)

    # Copy result back to host
    ses.copy_from(docker_so_name, so_name)
    return 0


if sys.platform == "darwin":

    def __create_shared_mac(so_name, objs, **kwargs):
        return link_shared_macos(so_name, objs, kwargs)

    create_shared = __create_shared_mac
    register_func("tvm.contrib.hexagon.link_shared", f=link_shared_macos, override=True)
else:  # Linux and Win32
    create_shared = cc.create_shared
    register_func("tvm.contrib.hexagon.link_shared", f=link_shared, override=True)


def create_aot_shared(so_name: Union[str, pathlib.Path], files, hexagon_arch: str, options=None):
    """Export Hexagon AOT module."""
    options = options or []
    if not os.access(str(HEXAGON_CLANG_PLUS), os.X_OK):
        raise Exception(
            'The Clang++ "' + str(HEXAGON_CLANG_PLUS) + '" does not exist or is not executable.'
        )
    if not HEXAGON_TOOLCHAIN:
        raise Exception(
            " The environment variable HEXAGON_TOOLCHAIN is unset. Please export "
            + "HEXAGON_TOOLCHAIN in your environment."
        )
    if not HEXAGON_SDK_ROOT:
        raise Exception(
            " The environment variable HEXAGON_SDK_ROOT is unset. Please export "
            + "HEXAGON_SDK_ROOT in your environment."
        )

    # The AOT C codegen uses TVM runtime functions
    # (e.g. TVMBackendAllocWorkspace) directly. On Hexagon these calls
    # should be made using functions pointers provided as __TVM*
    # variables in the provided context.  This workaround allows the
    # the TVM runtime symbols to be visible to the compiled shared
    # library.
    #
    # This workaround can be removed when AOT codegen can be done with
    # LLVM codegen.
    workaround_link_flags = os.environ.get("HEXAGON_SHARED_LINK_FLAGS")
    if workaround_link_flags:
        options.extend(workaround_link_flags.split())

    tvm_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / ".." / ".." / ".." / ".."
    compute_arch = f"compute{hexagon_arch}"
    compile_options = [
        f"-O3",
        f"-I{tvm_dir / 'include'}",
        f"-I{tvm_dir / '3rdparty' / 'dlpack' / 'include'}",
        f"-I{tvm_dir / '3rdparty' / 'dmlc-core' / 'include'}",
        f"-I{pathlib.Path(HEXAGON_SDK_ROOT) / 'rtos' / 'qurt' / compute_arch / 'include'/ 'posix'}",
        f"-I{pathlib.Path(HEXAGON_SDK_ROOT) / 'rtos' / 'qurt' / compute_arch / 'include' / 'qurt'}",
        f"-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
        f"-D_MACH_I32=int",
    ]

    # For debugging
    for path in HEXAGON_SDK_INCLUDE_DIRS:
        compile_options.append(f"-I{str(path)}")

    cross_compile = cc.cross_compiler(compile_func=hexagon_clang_plus())
    cross_compile.output_format = "o"
    c_files = [str(file) for file in files]
    cross_compile(str(so_name), c_files, options=compile_options + options)


def pack_imports(
    module: tvm.runtime.Module,
    is_system_lib: bool,  # pylint: disable=unused-argument
    c_symbol_prefix: str,
    workspace_dir: str,
):
    """Create an ELF object file that contains the binary data for the modules
    imported in `module`. This is a callback function for use as `fpack_imports`
    in `export_library`.

    Parameters
    ----------
    module: tvm.runtime.Module
        Module whose imported modules need to be serialized.
    is_system_lib: bool
        Flag whether the exported module will be used as a system library.
    c_symbol_prefix: str
        Prefix to prepend to the blob symbol.
    workspace_dir: str
        Location for created files.

    Returns
    -------
    file_name: str
        The name of the created object file.
    """

    path_bin = os.path.join(workspace_dir, "imports.bin")
    pack_to_bin_f_name = "runtime.ModulePackImportsToNDArray"
    fpack_to_bin = tvm.get_global_func(pack_to_bin_f_name)
    assert fpack_to_bin, f"Expecting {pack_to_bin_f_name} in registry"

    fpack_to_bin(module).numpy().tofile(path_bin)

    mblob_symbol = c_symbol_prefix + tvm.get_global_func("runtime.ModuleImportsBlobName")()

    binary_size = os.path.getsize(path_bin)
    hexagon_toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    assert hexagon_toolchain, "Please set HEXAGON_TOOLCHAIN variable"
    version = toolchain_version(hexagon_toolchain)
    assert (
        version[0] == 8 and version[1] >= 5
    ), "Please use Hexagon toolchain version 8.5.x or later"
    if version[1] <= 6:
        path_o = os.path.join(workspace_dir, f"{c_symbol_prefix}devc.o")
        subprocess.run(
            [
                f"{hexagon_toolchain}/bin/hexagon-clang",
                "-x",
                "c",
                "-c",
                "/dev/null",
                "-o",
                path_o,
            ],
            check=True,
        )
        subprocess.run(
            [
                f"{hexagon_toolchain}/bin/hexagon-llvm-objcopy",
                path_o,
                "--add-section",
                f".rodata={path_bin}",
                "--add-symbol",
                f"{mblob_symbol}=.rodata:0,object",
            ],
            check=True,
        )
        return path_o

    else:  # 8.6.07+
        path_c = os.path.join(workspace_dir, f"{c_symbol_prefix}devc.c")
        path_o = os.path.join(workspace_dir, f"{c_symbol_prefix}devc.o")
        with open(path_c, "w") as f:
            f.write(
                f"const unsigned char {mblob_symbol}[{binary_size}] "
                f'__attribute__((section(".rodata"))) = {{0x1}};'
            )
        subprocess.run(
            [f"{hexagon_toolchain}/bin/hexagon-clang", "-c", path_c, "-o", path_o], check=True
        )
        subprocess.run(
            [
                f"{hexagon_toolchain}/bin/hexagon-llvm-objcopy",
                path_o,
                "--update-section",
                f".rodata={path_bin}",
            ],
            check=True,
        )
        return path_o


def export_module(module, out_dir, binary_name="test_binary.so"):
    """Export Hexagon shared object to a file."""
    binary_path = pathlib.Path(out_dir) / binary_name
    module.save(str(binary_path))
    return binary_path


def allocate_hexagon_array(
    dev, tensor_shape=None, dtype=None, data=None, axis_separators=None, mem_scope=None
):
    """
    Allocate a hexagon array which could be a 2D array
    on physical memory defined by axis_separators
    """
    if tensor_shape is None:
        assert data is not None, "Must provide either tensor shape or numpy data array"
        tensor_shape = data.shape
    elif data is not None:
        assert (
            tensor_shape == data.shape
        ), "Mismatch between provided tensor shape and numpy data array shape"

    if dtype is None:
        assert data is not None, "Must provide either dtype or numpy data array"
        dtype = data.dtype.name
    elif data is not None:
        assert dtype == data.dtype, "Mismatch between provided dtype and numpy data array dtype"

    if axis_separators is None:
        axis_separators = []

    boundaries = [0, *axis_separators, len(tensor_shape)]
    physical_shape = [
        numpy.prod(tensor_shape[dim_i:dim_f])
        for dim_i, dim_f in zip(boundaries[:-1], boundaries[1:])
    ]

    arr = tvm.nd.empty(physical_shape, dtype=dtype, device=dev, mem_scope=mem_scope)

    if data is not None:
        arr.copyfrom(data.reshape(physical_shape))

    return arr._create_view(tensor_shape)


class ContainerSession:
    """Docker container session

    Parameters
    ----------
    base_image_name : str
        Docker image name to use. Empty string means to use default "tlcpack/ci-hexagon"
        base image.
    """

    def __init__(self, base_image_name: str = ""):
        self._client = None
        self._container = None
        self.tmp_dir = None

        self._client = ContainerSession._get_docker_client()

        if base_image_name == "":
            base_image_name = ContainerSession._get_latest_ci_image(self._client)

        self._container = ContainerSession._find_container_or_create(self._client, base_image_name)

        exit_code, tmp_dir_b = self._container.exec_run("mktemp -d -t tvm-toolbox-XXXXXXXXXX")
        assert exit_code == 0

        self.tmp_dir = tmp_dir_b.decode("utf-8").rstrip()

    def __del__(self):
        self.close()

    @staticmethod
    def _get_latest_ci_image(client) -> str:
        ci_images = client.images.list(name="tlcpack/ci-hexagon")
        ci_images.sort(reverse=True, key=lambda img: img.tags[0])
        return ci_images[0].tags[0]

    @staticmethod
    def _get_docker_client():
        try:
            # pylint: disable=import-outside-toplevel
            from docker import from_env
            from docker.errors import DockerException
        except (ModuleNotFoundError, ImportError):
            raise Exception("Docker SDK module is not installed. Please install it.")

        try:
            client = from_env()
        except DockerException:
            raise Exception(
                "Docker server is not available. Please verify the docker is installed, "
                "launched and available via command line ('dokcer ps' should works)."
            )

        return client

    @staticmethod
    def _find_container_or_create(client, image_name: str):
        all_containers = client.containers.list(all=True)

        filtered_containers = []
        for container in all_containers:
            tags: list = container.image.tags
            img_name: str = tags[0]
            if img_name.startswith(image_name) and container.name.startswith("tvm-hex-toolbox"):
                filtered_containers.append(container)

        if len(filtered_containers) == 0:
            container = client.containers.run(
                image=image_name, detach=True, tty=True, name="tvm-hex-toolbox"
            )
        else:
            container = filtered_containers[0]

        if container.status != "running":
            container.start()

        return container

    def exec(self, cmd) -> str:
        """Execute command inside docker container"""
        exit_code, res = self._container.exec_run(cmd)
        assert exit_code == 0
        return res.decode("utf-8")

    def get_env(self, key: str) -> str:
        """Return env var value from docker container"""
        res: str = self.exec(f"bash -c 'echo \"${key}\"'")
        return res.rstrip(" \n")

    def copy_to(self, host_file_path: str) -> str:
        """Upload file to docker container"""
        file_name = os.path.basename(host_file_path)

        byte_stream = io.BytesIO()
        with tarfile.open(fileobj=byte_stream, mode="w:gz") as tar:
            tar.add(host_file_path, arcname=file_name)

        self._container.put_archive(path=self.tmp_dir, data=byte_stream.getvalue())

        return f"{self.tmp_dir}/{file_name}"

    def copy_from(self, container_file_path: str, host_file_path: str):
        """Download file from docker container"""
        tar_bytes_gen, _ = self._container.get_archive(container_file_path)

        # convert to bytes
        tar_bytes = bytes()
        for chunk in tar_bytes_gen:
            tar_bytes += chunk

        tar = tarfile.open(fileobj=io.BytesIO(initial_bytes=tar_bytes))
        assert len(tar.getmembers()) == 1
        tar_element_reader = tar.extractfile(tar.getmembers()[0])
        with open(host_file_path, "wb") as host_file:
            for chunk in tar_element_reader:
                host_file.write(chunk)

    def close(self):
        """Close docker container session"""
        if self.tmp_dir is not None:
            exit_code, _ = self._container.exec_run(f"rm -rf {self.tmp_dir}")
            assert exit_code == 0
