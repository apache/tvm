"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging
import subprocess
import os

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from .cross_compile import create_lib


def init(device_type, runtime_lib_path=None, port=0):
    """Initializes a micro device context

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
        runtime_lib_path = get_init_lib(device_type)
    _MicroInit(device_type, runtime_lib_path, port)


def get_init_lib(device_type, src_path=None, cc=None):
    """Compiles code into a binary

    Parameters
    ----------
    device_type : str, optional
        type of low-level device

    src_path : str, optional
        path to source file

    cc : str, optional
        compiler command to be used

    Return
    ------
    obj_path : bytearray
        compiled binary file path
    """
    # use default init lib, if none is specified
    if src_path is None:
        micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                        "src", "runtime", "micro", "device")
        src_path = os.path.join(micro_device_dir, "utvm_runtime.cc")

    # choose compiler based on device type (if `cc` wasn't specified)
    if cc is None:
        if device_type == "host":
            cc = "gcc"
        elif device_type == "openocd":
            cc = "riscv-gcc"
        else:
            raise RuntimeError("unknown micro device type \"{}\"".format(device_type))

    obj_path = create_micro_lib(cc, src_path)
    return obj_path


def create_micro_lib(cc, src_path):
    """TODO"""
    obj_name = ".".join(os.path.basename(src_path).split(".")[:-1])
    obj_path = os.path.join(os.path.dirname(src_path), obj_name)
    options = ["-I" + path for path in find_include_path()] + ["-fno-stack-protector"]
    create_lib(obj_path, src_path, options, cc)
    return obj_path


_init_api("tvm.micro", "tvm.micro.base")
