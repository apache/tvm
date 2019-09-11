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
import sys

from tvm.contrib import util as _util
from tvm.contrib import cc as _cc

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path

SUPPORTED_DEVICE_TYPES = ["host", "openocd"]

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
      device_type = "openocd"
      toolchain_prefix = "riscv64-unknown-elf-"
      with tvm.micro.Session(device_type,
                             toolchain_prefix,
                             base_addr=0x10010000,
                             server_addr="127.0.0.1",
                             port=6666):
          c_mod.export_library(lib_obj_path, fcompile=tvm.micro.cross_compiler(toolchain_prefix))
          micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    """

    def __init__(self, device_type, toolchain_prefix, **kwargs):
        if device_type not in SUPPORTED_DEVICE_TYPES:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))
        self._check_system()
        self._check_args(device_type, kwargs)

        # First, find and compile runtime library.
        runtime_src_path = os.path.join(_get_micro_device_dir(), "utvm_runtime.c")
        tmp_dir = _util.tempdir()
        runtime_obj_path = tmp_dir.relpath("utvm_runtime.obj")
        create_micro_lib(
            runtime_obj_path, runtime_src_path, toolchain_prefix, include_dev_lib_header=False)

        base_addr = kwargs.get("base_addr", 0)
        server_addr = kwargs.get("server_addr", "")
        port = kwargs.get("port", 0)
        self.module = _CreateSession(
            device_type, runtime_obj_path, toolchain_prefix, base_addr, server_addr, port)
        self._enter = self.module["enter"]
        self._exit = self.module["exit"]

    def _check_system(self):
        """Check if the user's system is supported by MicroTVM.

        Raises error if not supported.
        """
        if not sys.platform.startswith("linux"):
            raise RuntimeError("microTVM is currently only supported on Linux")
        # TODO(weberlo): Add 32-bit support.
        # It's primarily the compilation pipeline that isn't compatible.
        if sys.maxsize <= 2**32:
            raise RuntimeError("microTVM is currently only supported on 64-bit platforms")

    def _check_args(self, device_type, args):
        """Check if the given configuration is valid."""
        if device_type == "host":
            pass
        elif device_type == "openocd":
            assert "base_addr" in args
            assert "server_addr" in args
            assert "port" in args

    def __enter__(self):
        self._enter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._exit()


def _get_micro_device_dir():
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

    include_dev_lib_header : Optional[bool]
        whether to include the device library header containing definitions of
        library functions.

    Return
    ------
    func : Callable[[str, str, Optional[str]], None]
        cross compile function taking a destination path for the object file
        and a path for the input source file.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      fcompile = tvm.micro.cross_compiler(toolchain_prefix="")
      c_mod.export_library("dev_lib.obj", fcompile=fcompile)
    """
    def compile_func(obj_path, src_path, **kwargs):
        if isinstance(obj_path, list):
            obj_path = obj_path[0]
        if isinstance(src_path, list):
            src_path = src_path[0]
        create_micro_lib(obj_path, src_path, toolchain_prefix,
                         kwargs.get("options", None), include_dev_lib_header)
    return _cc.cross_compiler(compile_func)


def create_micro_lib(
        obj_path, src_path, toolchain_prefix, options=None, include_dev_lib_header=True):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    obj_path : Optional[str]
        path to generated object file (defaults to same directory as `src_path`)

    src_path : str
        path to source file

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
    options += ["-I{}".format(_get_micro_device_dir())]
    options += ["-fno-stack-protector"]
    # TODO(weberlo): Don't rely on the toolchain prefix to identify if this is the host
    # device.
    if toolchain_prefix == "" and sys.maxsize > 2**32 and sys.platform.startswith("linux"):
        # Only add this option if the host is a 64-bit Linux.
        options += ["-mcmodel=large"]
    compile_cmd = "{}gcc".format(toolchain_prefix)

    if include_dev_lib_header:
        # Create a temporary copy of the source, so we can inject the dev lib
        # header without modifying the original.
        tmp_dir = _util.tempdir()
        temp_src_path = tmp_dir.relpath("temp.c")
        with open(src_path, "r") as f:
            src_lines = f.read().splitlines()
        src_lines.insert(0, "#include \"utvm_device_dylib_redirect.c\"")
        with open(temp_src_path, "w") as f:
            f.write("\n".join(src_lines))
        src_path = temp_src_path

    _cc.create_shared(obj_path, src_path, options, compile_cmd)


_init_api("tvm.micro", "tvm.micro.base")
