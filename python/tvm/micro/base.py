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

"""Base definitions for micro."""

from __future__ import absolute_import

import logging
import os

from tvm.contrib import util

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from .cross_compile import create_lib

SUPPORTED_DEVICE_TYPES = ["host"]

class Session:
    """MicroTVM Device Session

    Parameters
    ----------
    device_type : str
        type of low-level device

    toolchain_prefix : str
        toolchain prefix to be used. For example, a prefix of
        "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
        the compiler and "riscv64-unknown-elf-ld" is used as the linker,
        etc.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      device_type = "host"
      with tvm.micro.Session(device_type) as sess:
          sess.create_micro_mod(c_mod)
    """

    def __init__(self, device_type, toolchain_prefix):
        if device_type not in SUPPORTED_DEVICE_TYPES:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

        # First, find and compile runtime library.
        runtime_src_path = os.path.join(get_micro_device_dir(), "utvm_runtime.c")
        tmp_dir = util.tempdir()
        runtime_obj_path = tmp_dir.relpath("utvm_runtime.obj")
        create_micro_lib(
            runtime_src_path, runtime_obj_path, toolchain_prefix, include_dev_lib_header=False)

        self.module = _CreateSession(device_type, runtime_obj_path, toolchain_prefix)
        self._enter = self.module["enter"]
        self._exit = self.module["exit"]

    def __enter__(self):
        self._enter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._exit()


def get_micro_device_dir():
    """Get directory path for uTVM runtime source files.

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                    "src", "runtime", "micro", "device")
    return micro_device_dir


def cross_compiler(toolchain_prefix, include_dev_lib_header=True):
    """Creates a cross compile function that wraps `create_micro_lib`.

    For use in `tvm.module.Module.export_library`.

    Parameters
    ----------
    toolchain_prefix : str
        toolchain prefix to be used

    include_dev_lib_header : bool
        whether to include the device library header containing definitions of
        library functions.

    Return
    ------
    func : Callable[[str, str], None]
        cross compile function taking a destination path for the object file
        and a path for the input source file.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      fcompile = tvm.micro.cross_compiler(toolchain_prefix="")
      c_mod.export_library("dev_lib.obj", fcompile=fcompile)
    """
    def func(obj_path, src_path, **kwargs):
        if len(src_path) != 1:
            # Only a single source file can be used to generate an object
            # file with `gcc`.
            raise RuntimeError("multiple source files given to cross compiler")
        src_path = src_path[0]
        create_micro_lib(
            src_path, obj_path, toolchain_prefix, include_dev_lib_header=include_dev_lib_header)
    return func


def create_micro_lib(src_path, obj_path, toolchain_prefix, include_dev_lib_header=True):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    src_path : str
        path to source file

    obj_path : Optional[str]
        path to generated object file (defaults to same directory as `src_path`)

    toolchain_prefix : str
        toolchain prefix to be used

    include_dev_lib_header : bool
        whether to include the device library header containing definitions of
        library functions.
    """
    def replace_suffix(s, new_suffix):
        if "." in os.path.basename(s):
            # There already exists an extension.
            return os.path.join(
                os.path.dirname(s),
                ".".join(os.path.basename(s).split(".")[:-1] + [new_suffix]))
        # No existing extension; we can just append.
        return s + "." + new_suffix

    # uTVM object files cannot have an ".o" suffix, because it triggers the
    # code path for creating shared objects in `tvm.module.load`.  So we replace
    # ".o" suffixes with ".obj".
    if obj_path.endswith(".o"):
        logging.warning(
            "\".o\" suffix in \"%s\" has been replaced with \".obj\"", obj_path)
        obj_path = replace_suffix(obj_path, "obj")

    options = ["-I" + path for path in find_include_path()]
    options += ["-I{}".format(get_micro_device_dir())]
    options += ["-fno-stack-protector"]
    options += ["-mcmodel=large"]
    compile_cmd = "{}gcc".format(toolchain_prefix)

    if include_dev_lib_header:
        # Create a temporary copy of the source, so we can inject the dev lib
        # header without modifying the original.
        tmp_dir = util.tempdir()
        temp_src_path = tmp_dir.relpath("temp.c")
        with open(src_path, "r") as f:
            src_lines = f.read().splitlines()
        src_lines.insert(0, "#include \"utvm_device_dylib_redirect.c\"")
        with open(temp_src_path, "w") as f:
            f.write("\n".join(src_lines))
        create_lib(obj_path, temp_src_path, options, compile_cmd)
    else:
        # TODO(weberlo): Consolidate `create_lib` and `contrib.cc.cross_compiler`
        create_lib(obj_path, src_path, options, compile_cmd)


_init_api("tvm.micro", "tvm.micro.base")
