# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Utility to invoke nvcc compiler in the system"""
from __future__ import absolute_import as _abs

import subprocess
import os
import warnings

import tvm._ffi
from tvm.runtime import ndarray as nd

from . import utils
from .._ffi.base import py_str


def compile_cuda(code, target="ptx", arch=None, options=None, path_target=None):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target : str
        The target format

    arch : str
        The architecture

    options : str or list of str
        The additional options

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    temp = utils.tempdir()
    if target not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target must be in cubin, ptx, fatbin")
    temp_code = temp.relpath("my_kernel.cu")
    temp_target = temp.relpath("my_kernel.%s" % target)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    if arch is None:
        if nd.cuda(0).exist:
            # auto detect the compute arch argument
            arch = "sm_" + "".join(nd.cuda(0).compute_version.split("."))
        else:
            raise ValueError("arch(sm_xy) is not passed, and we cannot detect it from env")

    file_target = path_target if path_target else temp_target
    cmd = ["nvcc"]
    cmd += ["--%s" % target, "-O3"]
    if isinstance(arch, list):
        cmd += arch
    else:
        cmd += ["-arch", arch]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    # NOTE: ccbin option can be used to tell nvcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require nvcc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    data = bytearray(open(file_target, "rb").read())
    if not data:
        raise RuntimeError("Compilation error: empty result is generated")
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(str(out).strip(), "../.."))
    cuda_path = "/usr/local/cuda"
    if os.path.exists(os.path.join(cuda_path, "bin/nvcc")):
        return cuda_path
    raise RuntimeError("Cannot find cuda path")


def get_cuda_version(cuda_path):
    """Utility function to get cuda version

    Parameters
    ----------
    cuda_path : str
        Path to cuda root.

    Returns
    -------
    version : float
        The cuda version
    """
    version_file_path = os.path.join(cuda_path, "version.txt")
    if not os.path.exists(version_file_path):
        # Debian/Ubuntu repackaged CUDA path
        version_file_path = os.path.join(cuda_path, "lib", "cuda", "version.txt")
    try:
        with open(version_file_path) as f:
            version_str = f.readline().replace("\n", "").replace("\r", "")
            return float(version_str.split(" ")[2][:2])
    except FileNotFoundError:
        pass

    cmd = [os.path.join(cuda_path, "bin", "nvcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "release" in l][0]
        release_fields = [s.strip() for s in release_line.split(",")]
        release_version = [f[1:] for f in release_fields if f.startswith("V")][0]
        major_minor = ".".join(release_version.split(".")[:2])
        return float(major_minor)
    raise RuntimeError("Cannot read cuda version file")


@tvm._ffi.register_func("tvm_callback_libdevice_path")
def find_libdevice_path(arch):
    """Utility function to find libdevice

    Parameters
    ----------
    arch : int
        The compute architecture in int

    Returns
    -------
    path : str
        Path to libdevice.
    """
    cuda_path = find_cuda_path()
    lib_path = os.path.join(cuda_path, "nvvm/libdevice")
    if not os.path.exists(lib_path):
        # Debian/Ubuntu repackaged CUDA path
        lib_path = os.path.join(cuda_path, "lib/nvidia-cuda-toolkit/libdevice")
    selected_ver = 0
    selected_path = None
    cuda_ver = get_cuda_version(cuda_path)
    if cuda_ver in (9.0, 9.1, 10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 11.3):
        path = os.path.join(lib_path, "libdevice.10.bc")
    else:
        for fn in os.listdir(lib_path):
            if not fn.startswith("libdevice"):
                continue
            ver = int(fn.split(".")[-3].split("_")[-1])
            if selected_ver < ver <= arch:
                selected_ver = ver
                selected_path = fn
        if selected_path is None:
            raise RuntimeError("Cannot find libdevice for arch {}".format(arch))
        path = os.path.join(lib_path, selected_path)
    return path


def callback_libdevice_path(arch):
    try:
        return find_libdevice_path(arch)
    except RuntimeError:
        warnings.warn("Cannot find libdevice path")
        return ""


def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the arch in three different places, first in the target attributes, then the global
    scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    # 1. Target
    if target:
        if "arch" in target.attrs:
            compute_version = target.attrs["arch"]
            major, minor = compute_version.split("_")[1]
            return major + "." + minor

    # 2. Global scope
    from tvm.autotvm.env import AutotvmGlobalScope  # pylint: disable=import-outside-toplevel

    if AutotvmGlobalScope.current.cuda_target_arch:
        major, minor = AutotvmGlobalScope.current.cuda_target_arch.split("_")[1]
        return major + "." + minor

    # 3. GPU
    if tvm.cuda(0).exist:
        return tvm.cuda(0).compute_version

    warnings.warn(
        "No CUDA architecture was specified or GPU detected."
        "Try specifying it by adding '-arch=sm_xx' to your target."
    )
    return None


def parse_compute_version(compute_version):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error: " + str(err))


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, minor = parse_compute_version(compute_version)
    # fp 16 support in reference to:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions
    if major == 5 and minor == 3:
        return True
    if major >= 6:
        return True

    return False


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 6:
        return True

    return False


def have_tensorcore(compute_version=None, target=None):
    """Either TensorCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    target : tvm.target.Target, optional
        The compilation target, will be used to determine arch if compute_version
        isn't specified.
    """
    if compute_version is None:
        if tvm.cuda(0).exist:
            compute_version = tvm.cuda(0).compute_version
        else:
            if target is None or "arch" not in target.attrs:
                warnings.warn(
                    "Tensorcore will be disabled due to no CUDA architecture specified."
                    "Try specifying it by adding '-arch=sm_xx' to your target."
                )
                return False
            compute_version = target.attrs["arch"]
            # Compute version will be in the form "sm_{major}{minor}"
            major, minor = compute_version.split("_")[1]
            compute_version = major + "." + minor
    major, _ = parse_compute_version(compute_version)
    if major >= 7:
        return True

    return False


def have_cudagraph():
    """Either CUDA Graph support is provided"""
    try:
        cuda_path = find_cuda_path()
        cuda_ver = get_cuda_version(cuda_path)
        if cuda_ver < 10.0:
            return False
        return True
    except RuntimeError:
        return False


def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_compute_version(compute_version)
    if major >= 8:
        return True

    return False
