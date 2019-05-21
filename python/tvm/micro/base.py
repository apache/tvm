"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging
import subprocess
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


def from_host_mod(host_mod, device_type):
    """Produces a micro module from a given host module.

    Parameters
    ----------
    host_mod : tvm.module.Module
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
    mod_src = host_mod.get_source()
    with open(lib_src_path, "w") as f:
        f.write(mod_src)
    # Compile to object file.
    lib_obj_path = create_micro_lib(lib_src_path, device_type)
    micro_mod = tvm.module.load(lib_obj_path, "micro_dev")
    return micro_mod


def create_micro_lib(src_path, device_type, cc=None):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    src_path : str
        path to source file

    device_type : str
        type of low-level device

    cc : str, optional
        compiler command to be used

    Return
    ------
    obj_path : bytearray
        compiled binary file path
    """
    # Choose compiler based on device type (if `cc` wasn't specified).
    if cc is None:
        if device_type == "host":
            cc = "gcc"
        elif device_type == "openocd":
            cc = "riscv-gcc"
        else:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

    obj_name = ".".join(os.path.basename(src_path).split(".")[:-1])
    obj_path = os.path.join(os.path.dirname(src_path), obj_name)
    options = ["-I" + path for path in find_include_path()] + ["-fno-stack-protector"]
    create_lib(obj_path, src_path, options, cc)
    return obj_path


_init_api("tvm.micro", "tvm.micro.base")
