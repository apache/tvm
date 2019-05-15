"""Cross compilation for micro."""

from __future__ import absolute_import

import struct
import logging
import subprocess
import os

from .._ffi.function import _init_api
from .._ffi.base import py_str


def create_lib(output, sources, options=None, cc="gcc"):
    """Compiles source code into a binary object file

    Parameters
    ----------
    output : str
        target library path

    sources : list
        list of source files to be compiled

    options: list
        list of additional option strings

    cc : str, optional
        compiler string
    """
    cmd = [cc]
    cmd += ["-x", "c", "-c"]
    cmd += ["-o", output]
    if isinstance(sources, str):
        cmd += [sources]
    else:
        cmd += sources
    if options:
        cmd += options
    print(f"compiling with command \"{cmd}\"")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Error in compilation:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


_init_api("tvm.micro.cc")
