# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""
from __future__ import absolute_import as _abs
import sys
import subprocess
from . import util

def compile_cuda(code, target="ptx", arch=None,
                 options=None, path_target=None):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target : str
        The target format

    arch : str
        The architecture

    options : str
        The additional options

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    temp = util.tempdir()
    if target not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target must be in cubin, ptx, fatbin")
    temp_code = temp.relpath("my_kernel.cu")
    temp_target = temp.relpath("my_kernel.%s" % target)

    with open(temp_code, "w") as out_file:
        out_file.write(code)
    if target == "cubin" and arch is None:
        raise ValueError("arch(sm_xy) must be passed for generating cubin")

    file_target = path_target if path_target else temp_target
    cmd = ["nvcc"]
    cmd += ["--%s" % target, "-O3"]
    if arch:
        cmd += ["-arch", arch]
    cmd += ["-o", file_target]

    if options:
        cmd += options
    cmd += [temp_code]
    args = ' '.join(cmd)

    proc = subprocess.Popen(
        args, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(str(out))
        sys.stderr.flush()
        cubin = None
    else:
        cubin = bytearray(open(file_target, "rb").read())
    return cubin
