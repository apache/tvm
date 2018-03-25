# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""
from __future__ import absolute_import as _abs

import subprocess
import os
import warnings
from . import util
from .. import ndarray as nd
from ..api import register_func
from .._ffi.base import py_str

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
        msg += py_str(out)
        raise RuntimeError(msg)

    data = bytearray(open(file_target, "rb").read())
    if not data:
        raise RuntimeError(
            "Compilation error: empty result is generated")
    return data

def find_cuda_path():
    """Utility function to find cuda path

    Returns
    -------
    path : str
        Path to cuda root.
    """
    if "CUDA_PATH" in os.environ:
        return os.environ["CUDA_PATH"]
    cmd = ["which", "nvcc"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        return os.path.abspath(os.path.join(str(out).strip(), "../.."))
    cuda_path = "/usr/local/cuda"
    if os.path.exists(os.path.join(cuda_path, "bin/nvcc")):
        return cuda_path
    raise RuntimeError("Cannot find cuda path")


@register_func("tvm_callback_libdevice_path")
def find_libdevice_path(arch):
    """Utility function to find libdevice

    Parameters
    ----------
    arch : int
        The compute architecture in int
    """
    cuda_path = find_cuda_path()
    lib_path = os.path.join(cuda_path, "nvvm/libdevice")
    selected_ver = 0
    selected_path = None

    for fn in os.listdir(lib_path):
        if not fn.startswith("libdevice"):
            continue
        ver = int(fn.split(".")[-3].split("_")[-1])
        if ver > selected_ver and ver <= arch:
            selected_ver = ver
            selected_path = fn
    if selected_path is None:
        raise RuntimeError("Cannot find libdevice for arch {}".format(arch))
    return os.path.join(lib_path, selected_path)


def callback_libdevice_path(arch):
    try:
        return find_libdevice_path(arch)
    except RuntimeError:
        warnings.warn("Cannot find libdevice path")
        return ""
