"""Utility for Interacting with SDAccel Tools"""
import subprocess
import os
from . import util
from ..api import register_func


@register_func("tvm_callback_sdaccel_compile")
def compile_vhls(codes, kernels):
    """Compile Vivado HLS code for SDAccel.

    Parameters
    ----------
    codes : str
        Array of the Vivado HLS codes.

    kernels : str
        Array of the kernels to compile or link.

    Return
    ------
    xclbin : bytearray
        The bytearray of the xclbin
    """
    tmp_dir = util.tempdir()

    sdk = os.environ.get("XILINX_SDX", None)
    xocc = os.path.join(sdk, "bin/xocc") if sdk else "xocc"
    target = os.environ.get("XCL_TARGET",
                            "sw_emu" if os.environ.get("XCL_EMULATION_MODE") else "hw")
    advanced_params = ["--xp", "param:compiler.preserveHlsOutput=1",
                       "--xp", "param:compiler.generateExtraRunData=true"]
    platform = os.environ.get("XCL_PLATFORM", os.environ.get("AWS_PLATFORM"))

    if platform is None:
        raise RuntimeError("No Xlinx device specified.")

    tmp_xo_files = []
    for code, kernel in zip(codes, kernels):
        code = code.value
        kernel = kernel.value

        tmp_cpp = tmp_dir.relpath(kernel + ".cpp")
        tmp_xo = tmp_dir.relpath(kernel + ".xo")

        with open(tmp_cpp, "wb") as out_file:
            out_file.write(bytes(code))

        # build xo
        args = [xocc, "-c", "-t", target, "--platform", platform, "-o", tmp_xo, "-k", kernel] + \
               advanced_params + [tmp_cpp]
        returncode = subprocess.call(args)
        if returncode != 0:
            raise RuntimeError("Compile error")

        tmp_xo_files.append(tmp_xo)

    # build xclbin
    tmp_xclbin = tmp_dir.relpath("output.xclbin")
    args = [xocc, "-l", "-t", target, "--platform", platform, "-o", tmp_xclbin] + tmp_xo_files + \
           advanced_params
    returncode = subprocess.call(args)
    if returncode != 0:
        raise RuntimeError("Link error")

    return bytearray(open(tmp_xclbin, "rb").read())
