# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""
from __future__ import absolute_import as _abs

import subprocess
from . import util
from .. import ndarray as nd

def compile_cuda(code,
                 target="ptx",
                 arch=None,
                 options=None,
                 path_target=None):
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

    if arch is None:
        if nd.gpu(0).exist:
            # auto detect the compute arch argument
            arch = "sm_" + "".join(nd.gpu(0).compute_version.split('.'))
        else:
            raise ValueError("arch(sm_xy) is not passed, and we cannot detect it from env")

    file_target = path_target if path_target else temp_target
    cmd = ["nvcc"]
    cmd += ["--%s" % target, "-O3"]
    cmd += ["-arch", arch]
    cmd += ["-o", file_target]

    if options:
        cmd += options
    cmd += [temp_code]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += out
        raise RuntimeError(msg)

    return bytearray(open(file_target, "rb").read())
