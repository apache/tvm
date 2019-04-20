"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging
import subprocess
from os.path import join, dirname

from .._ffi.function import _init_api
from .._ffi.base import py_str
from .._ffi.libinfo import find_include_path
from ..contrib import util
from ..api import register_func, convert


def micro_init(device_type, init_source, port=0):
    """Compiles code into a binary

    Parameters
    ----------
    device_type : str
        type of low-level device

    init_binary_path : str
        path to init stub binary

    port : integer
        port number of OpenOCD server 
    """
    _MicroInit(device_type, init_source, port)


def get_init_lib(source_path, device_type="", cc="gcc"):
    """Compiles code into a binary

    Parameters
    ----------
    source_path : str
        path to source file

    device_type : str
        type of low-level device

    cc : str
        compiler to be used

    Return
    ------
    obj_path : bytearray
        compiled binary file path
    """
    if device_type == "host":
        cc = "gcc"
    elif device_type == "openocd":
        cc = "riscv-gcc"
    obj_path = join(dirname(source_path), "utvm_runtime.o")
    includes = ["-I" + path for path in find_include_path()]
    options = ["-fno-stack-protector"]
    cmd = [cc, "-x", "c", "-c", "-o", obj_path, source_path]
    cmd += includes
    cmd += options
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Error in compilation:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    return obj_path


_init_api("tvm.micro", "tvm.micro.base")
