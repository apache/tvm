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
from tvm.contrib import util

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from .cross_compile import create_lib

def init(device_type, runtime_lib_path=None, port=0):
    """Initializes a micro device context.

    Parameters
    ----------
    device_type : str
        type of low-level device

    runtime_lib_path : str, optional
        path to runtime lib binary

    port : integer, optional
        port number of OpenOCD server
    """
    if runtime_lib_path is None:
        # Use default init lib, if none is specified.
        micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                        "src", "runtime", "micro", "device")
        src_path = os.path.join(micro_device_dir, "utvm_runtime.c")
        runtime_lib_path = create_micro_lib(src_path, device_type)
    _MicroInit(device_type, runtime_lib_path, port)


def from_source_module(mod, device_type):
    """Produces a micro module from a given module.

    Parameters
    ----------
    mod : tvm.module.Module
        module for host execution

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
    mod_src = mod.get_source()
    with open(lib_src_path, "w") as f:
        f.write(mod_src)
    # Compile to object file.
    lib_obj_path = create_micro_lib(lib_src_path, device_type)
    micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    return micro_mod


def create_micro_lib(src_path, device_type, compile_cmd=None, obj_path=None):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    src_path : str
        path to source file

    device_type : str
        type of low-level device

    compile_cmd : str, optional
        compiler command to be used

    obj_path : str, optional
        path to generated object file (defaults to same directory as
        `src_path`)

    Return
    ------
    obj_path : bytearray
        compiled binary file path
    """
    # Choose compiler based on device type (if `compile_cmd` wasn't specified).
    if compile_cmd is None:
        if device_type == "host":
            compile_cmd = "gcc"
        elif device_type == "openocd":
            compile_cmd = "riscv-gcc"
        else:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

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
            "\".o\" suffix in \"%s\" has been replaced with \".obj\"" % obj_path)
        obj_path = replace_suffix(obj_path, "obj")

    options = ["-I" + path for path in find_include_path()] + ["-fno-stack-protector"]
    # TODO(weberlo): Consolidate `create_lib` and `contrib.cc.cross_compiler`
    create_lib(obj_path, src_path, options, compile_cmd)
    return obj_path


_init_api("tvm.micro", "tvm.micro.base")
