"""Utility for Interacting with SPIRV Tools"""
import subprocess
import os
from . import util
from .._ffi.base import py_str

def optimize(spv_bin):
    """Optimize SPIRV using spirv-opt via CLI

    Note that the spirv-opt is still experimental.

    Parameters
    ----------
    spv_bin : bytearray
        The spirv file

    Return
    ------
    cobj_bin : bytearray
        The HSA Code Object
    """

    tmp_dir = util.tempdir()
    tmp_in = tmp_dir.relpath("input.spv")
    tmp_out = tmp_dir.relpath("output.spv")
    with open(tmp_in, "wb") as out_file:
        out_file.write(bytes(spv_bin))

    sdk = os.environ.get("VULKAN_SDK", None)
    cmd = os.path.join(sdk, "bin/spirv-opt") if sdk else "spirv-opt"
    args = [cmd, "-O", tmp_in, "-o", tmp_out]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Opitmizationerror using spirv-opt:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    return bytearray(open(tmp_out, "rb").read())
