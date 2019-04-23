"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging
import subprocess
import os

from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from .cc import create_lib


def init(device_type, runtime_lib_path, port=0):
    """Compiles code into a binary

    Parameters
    ----------
    device_type : str
        type of low-level device

    runtime_lib_path : str
        path to runtime lib binary

    port : integer, optional
        port number of OpenOCD server 
    """
    _MicroInit(device_type, runtime_lib_path, port)


def get_init_lib(source_path="", device_type="", cc="gcc"):
    """Compiles code into a binary

    Parameters
    ----------
    source_path : str, optional
        path to source file

    device_type : str, optional
        type of low-level device

    cc : str, optional
        compiler to be used

    Return
    ------
    obj_path : bytearray
        compiled binary file path
    """
    if source_path == "":
        micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                        "src", "runtime", "micro", "device")
        sources = os.path.join(micro_device_dir, "utvm_runtime.cc")
    if device_type == "host":
        cc = "gcc"
    elif device_type == "openocd":
        cc = "riscv-gcc"
    output = os.path.join(os.path.dirname(source_path), "utvm_runtime.o")
    options = ["-I" + path for path in find_include_path()] + ["-fno-stack-protector"]
    create_lib(output, sources, options, cc)
    return output


_init_api("tvm.micro", "tvm.micro.base")
