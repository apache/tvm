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
# pylint: disable=invalid-name
"""Tools/compilers/linkers for Hexagon"""

import os
import pathlib
from typing import Union
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
            'hex_arch' - Hexagon architecture, e.g. v66
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
    hex_arch = extra_args.get("hex_arch") or "v66"
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
