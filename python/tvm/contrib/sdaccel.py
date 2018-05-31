"""Utility for Interacting with SDAccel Tools"""
import subprocess
import os
from . import util
from ..api import register_func

@register_func("tvm_callback_sdaccel_compile")
def compile_vhls(code, kernel):
    """Compile Vivado HLS code for SDAccel.

    Parameters
    ----------
    code : str
        The Vivado HLS code.

    kernel : str
        The kernel to compile or link.

    Return
    ------
    xclbin : bytearray
        The bytearray of the xclbin
    """
    tmp_dir = util.tempdir()
    tmp_cpp = tmp_dir.relpath("input.cpp")
    tmp_xo = tmp_dir.relpath("output.xo")
    tmp_xclbin = tmp_dir.relpath("output.xclbin")

    with open(tmp_cpp, "wb") as out_file:
        out_file.write(bytes(code))

    sdk = os.environ.get("XILINX_SDX", None)
    xocc = os.path.join(sdk, "bin/xocc") if sdk else "xocc"
    target = os.environ.get("XCL_TARGET",
                            "sw_emu" if os.environ.get("XCL_EMULATION_MODE") else "hw")
    advanced_params = ["--xp", "param:compiler.preserveHlsOutput=1",
                       "--xp", "param:compiler.generateExtraRunData=true"]
    platform = os.environ.get("XCL_PLATFORM",
                              os.environ.get("AWS_PLATFORM", "xilinx:kcu1500:dynamic"))
    # build xo
    args = [xocc, "-c", "-t", target, "--platform", platform, "-o", tmp_xo, "-k", kernel] + \
           advanced_params + [tmp_cpp]
    returncode = subprocess.call(args)
    if returncode != 0:
        raise RuntimeError("Compile error")

    # build xclbin
    args = [xocc, "-l", "-t", target, "--platform", platform, "-o", tmp_xclbin, tmp_xo] + \
           advanced_params
    returncode = subprocess.call(args)
    if returncode != 0:
        raise RuntimeError("Link error")

    return bytearray(open(tmp_xclbin, "rb").read())
