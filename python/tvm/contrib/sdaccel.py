"""Utility for Interacting with SDAccel Tools"""
import subprocess
import os
import re
from . import util
from ..api import register_func


def _vhls_to_opencl(code):
    """Convert source code from Vivado HLS to OpenCL."""
    out = ''
    for line in code.split('\n'):
        if re.match(r'#include', line):
            # OpenCL doesn't support include.
            continue
        if re.match(r'#pragma', line):
            # Remove Vivado HLS specific pragmas.
            continue

        if re.match(r'extern "C"', line):
            line = re.sub(r'^extern "C"', "__kernel", line)
            # Add __global to pointer parameters.
            line = re.sub(r'(\w+)\s*\*', r"__global \1*", line)

        out += line + '\n'

    return out


def _fake_compile_vhls(code):
    """Fake compile Vivado HLS code for SDAccel.

    Compile the Vivado HLS code as an OpenCL code, and generate a program
    binary for GPU which can be used instead of xclbin.

    Parameters
    ----------
    code : str
        The Vivado HLS code.

    Return
    ------
    binary : bytearray
        The program binary which can be passed to clCreateProgramWithBinary
    """
    try:
        import pyopencl as cl
    except ImportError:
        raise ImportError('PyOpenCL is required for testing SDAccel backend.')
    ctx = cl.Context(dev_type=cl.device_type.GPU)
    program = cl.Program(ctx, _vhls_to_opencl(code)).build()
    binary = bytearray(program.binaries[0])
    return binary


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
    platform = os.environ.get("XCL_PLATFORM", os.environ.get("AWS_PLATFORM"))

    if platform is None:
        # If we don't have the Xilinx toolchain, create a program binary for
        # GPU and use it for testing.
        return _fake_compile_vhls(code)

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
