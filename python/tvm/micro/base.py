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

import tvm.module
from tvm.contrib import graph_runtime, util
from tvm import relay

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from .cross_compile import create_lib

SUPPORTED_DEVICE_TYPES = ["host", "openocd"]

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

    def __init__(self, device_type, binutil_prefix, port=0):
        """Stores parameters for initializing a micro device session.

        The session is not initialized until the constructed object is used
        in a `with` block.

        Parameters
        ----------
        device_type : str
            type of low-level device

        binutil_prefix : str
            binutil prefix to be used. For example, a prefix of
            "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
            the compiler and "riscv64-unknown-elf-ld" is used as the linker,
            etc.

        port : integer, optional
            port number of OpenOCD server
        """
        if device_type not in SUPPORTED_DEVICE_TYPES:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

        self.device_type = device_type
        self.binutil_prefix = binutil_prefix
        self.port = port

    def micro_build(self, func: relay.Function, params={}):
        """Create a graph runtime module with a micro device context."""
        with tvm.build_config(disable_vectorize=True):
            with relay.build_config(opt_level=3):
                graph, c_mod, params = relay.build(func, target="c", params=params)

        micro_mod = self.create_micro_mod(c_mod)
        ctx = tvm.micro_dev(0)
        mod = graph_runtime.create(graph, micro_mod, ctx)
        return mod, params

    def create_micro_mod(self, c_mod):
        """Produces a micro module from a given module.

        Parameters
        ----------
        c_mod : tvm.module.Module
            module with "c" as its target backend

        device_type : str
            type of low-level device to target

        Return
        ------
        micro_mod : tvm.module.Module
            micro module for the target device
        """
        temp_dir = util.tempdir()
        # Save module source to temp file.
        lib_src_path = temp_dir.relpath("dev_lib.c")
        mod_src = c_mod.get_source()
        with open(lib_src_path, "w") as f:
            f.write(mod_src)
        # Compile to object file.
        lib_obj_path = self.create_micro_lib(lib_src_path)
        micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
        return micro_mod

    def create_micro_lib(self, src_path, obj_path=None):
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
        create_lib(obj_path, src_path, options, self._compile_cmd())
        return obj_path

    def _compile_cmd(self):
        return "{}gcc".format(self.binutil_prefix)

    def __enter__(self):
        # First, find and compile runtime library.
        micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                        "src", "runtime", "micro", "device")
        runtime_src_path = os.path.join(micro_device_dir, "utvm_runtime.c")
        tmp_dir = util.tempdir()
        runtime_lib_path = tmp_dir.relpath("utvm_runtime.obj")
        runtime_lib_path = self.create_micro_lib(runtime_src_path, obj_path=runtime_lib_path)

        # Then, initialize the session (includes loading the compiled runtime lib).
        _InitSession(self.device_type, runtime_lib_path, self.port)

        # Return `self` to bind the session as a variable in the `with` block.
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _EndSession()


_init_api("tvm.micro", "tvm.micro.base")
