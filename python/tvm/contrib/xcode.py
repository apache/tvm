# pylint: disable=invalid-name
"""Utility to invoke Xcode compiler toolchain"""
from __future__ import absolute_import as _abs
import sys
import subprocess
from . import util

def compile_metal(code, path_target=None, sdk="macosx"):
    """Compile metal with CLI tool from env.

    Parameters
    ----------
    code : str
        The cuda code.

    path_target : str, optional
        Output file.

    sdk : str, optional
        The target platform SDK.

    Return
    ------
    metallib : bytearray
        The bytearray of the metallib
    """
    temp = util.tempdir()
    temp_code = temp.relpath("my_lib.metal")
    temp_ir = temp.relpath("my_lib.air")
    temp_target = temp.relpath("my_lib.metallib")

    with open(temp_code, "w") as out_file:
        out_file.write(code)
    file_target = path_target if path_target else temp_target

    cmd1 = ["xcrun", "-sdk", sdk, "metal", "-O3"]
    cmd1 += [temp_code, "-o", temp_ir]
    cmd2 = ["xcrun", "-sdk", sdk, "metallib"]
    cmd2 += [temp_ir, "-o", file_target]
    proc = subprocess.Popen(
        ' '.join(cmd1) + ";" + ' '.join(cmd2),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(out)
        sys.stderr.flush()
        libbin = None
    else:
        libbin = bytearray(open(file_target, "rb").read())
    return libbin
