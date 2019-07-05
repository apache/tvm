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
    """MicroTVM Session

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      device_type = "host"
      with tvm.micro.Session(device_type) as sess:
          sess.create_micro_mod(c_mod)
    """

    def __init__(self, device_type, toolchain_prefix):
        """Stores parameters for initializing a micro device session.

        The session is not initialized until the constructed object is used
        in a `with` block.

        Parameters
        ----------
        device_type : str
            type of low-level device

        toolchain_prefix : str
            toolchain prefix to be used. For example, a prefix of
            "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
            the compiler and "riscv64-unknown-elf-ld" is used as the linker,
            etc.
        """
        if device_type not in SUPPORTED_DEVICE_TYPES:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

        # First, find and compile runtime library.
        micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                        "src", "runtime", "micro", "device")
        runtime_src_path = os.path.join(micro_device_dir, "utvm_runtime.c")
        tmp_dir = util.tempdir()
        runtime_lib_path = tmp_dir.relpath("utvm_runtime.obj")
        runtime_lib_path = create_micro_lib(
            runtime_src_path, toolchain_prefix, obj_path=runtime_lib_path)

        self.module = _CreateSession(device_type, runtime_lib_path, toolchain_prefix)
        self._enter = self.module["enter"]

    def __enter__(self):
        self._enter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


def create_micro_lib(src_path, toolchain_prefix, obj_path=None):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    src_path : str
        path to source file

    obj_path : str, optional
        path to generated object file (defaults to same directory as
        `src_path`)

    Return
    ------
    obj_path : bytearray
        compiled binary file path (will match input `obj_path`, if it was specified)
    """
    def replace_suffix(s, new_suffix):
        if "." in os.path.basename(s):
            # There already exists an extension.
            return os.path.join(
                os.path.dirname(s),
                ".".join(os.path.basename(s).split(".")[:-1] + [new_suffix]))
        # No existing extension; we can just append.
        return s + "." + new_suffix

    if obj_path is None:
        obj_name = replace_suffix(src_path, "obj")
        obj_path = os.path.join(os.path.dirname(src_path), obj_name)
    # uTVM object files cannot have an ".o" suffix, because it triggers the
    # code path for creating shared objects in `tvm.module.load`.  So we replace
    # ".o" suffixes with ".obj".
    if obj_path.endswith(".o"):
        logging.warning(
            "\".o\" suffix in \"%s\" has been replaced with \".obj\"", obj_path)
        obj_path = replace_suffix(obj_path, "obj")

    options = ["-I" + path for path in find_include_path()] + ["-fno-stack-protector"]
    # TODO(weberlo): Consolidate `create_lib` and `contrib.cc.cross_compiler`
    create_lib(obj_path, src_path, options, "{}gcc".format(toolchain_prefix))
    return obj_path


_init_api("tvm.micro", "tvm.micro.base")
