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
"""Utility for ROCm backend"""
import re
import subprocess
import os
from os.path import join, exists

import tvm._ffi
from tvm._ffi.base import py_str
import tvm.runtime
import tvm.target

from . import utils


def find_lld(required=True):
    """Find ld.lld in system.

    Parameters
    ----------
    required : bool
        Whether it is required,
        runtime error will be raised if the compiler is required.

    Returns
    -------
    valid_list : list of str
        List of possible paths.

    Note
    ----
    This function will first search ld.lld that
    matches the major llvm version that built with tvm
    """
    lld_list = []
    major = tvm.target.codegen.llvm_version_major(allow_none=True)
    if major is not None:
        lld_list += [f"ld.lld-{major}.0"]
        lld_list += [f"ld.lld-{major}"]
    lld_list += ["ld.lld"]
    valid_list = [utils.which(x) for x in lld_list]
    valid_list = [x for x in valid_list if x]
    if not valid_list and required:
        raise RuntimeError("cannot find ld.lld, candidates are: " + str(lld_list))
    return valid_list


def rocm_link(in_file, out_file, lld=None):
    """Link relocatable ELF object to shared ELF object using lld

    Parameters
    ----------
    in_file : str
        Input file name (relocatable ELF object file)

    out_file : str
        Output file name (shared ELF object file)

    lld : str, optional
        The lld linker, if not specified,
        we will try to guess the matched clang version.
    """

    # if our result has undefined symbols, it will fail to load
    # (hipModuleLoad/hipModuleLoadData), but with a somewhat opaque message
    # so we have ld.lld check this here.
    # If you get a complaint about missing symbols you might want to check the
    # list of bitcode files below.
    args = [
        lld if lld is not None else find_lld()[0],
        "--no-undefined",
        "-shared",
        in_file,
        "-o",
        out_file,
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Linking error using ld.lld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


@tvm._ffi.register_func("tvm_callback_rocm_link")
def callback_rocm_link(obj_bin):
    """Links object file generated from LLVM to HSA Code Object

    Parameters
    ----------
    obj_bin : bytearray
        The object file

    Return
    ------
    cobj_bin : bytearray
        The HSA Code Object
    """
    tmp_dir = utils.tempdir()
    tmp_obj = tmp_dir.relpath("rocm_kernel.o")
    tmp_cobj = tmp_dir.relpath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    rocm_link(tmp_obj, tmp_cobj)
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin


@tvm._ffi.register_func("tvm_callback_rocm_bitcode_path")
def callback_rocm_bitcode_path(rocdl_dir=None):
    """Utility function to find ROCm device library bitcodes

    Parameters
    ----------
    rocdl_dir : str
        The path to rocm library directory
        The default value is the standard location
    """
    # seems link order matters.

    if rocdl_dir is None:
        if exists("/opt/rocm/amdgcn/bitcode/"):
            rocdl_dir = "/opt/rocm/amdgcn/bitcode/"  # starting with rocm 3.9
        else:
            rocdl_dir = "/opt/rocm/lib/"  # until rocm 3.8

    bitcode_names = [
        "oclc_daz_opt_on",
        "ocml",
        "irif",  # this does not exist in rocm 3.9, drop eventually
        "oclc_correctly_rounded_sqrt_off",
        "oclc_correctly_rounded_sqrt_on",
        "oclc_daz_opt_off",
        "oclc_finite_only_off",
        "oclc_finite_only_on",
        # todo (t-vi): an alternative might be to scan for the
        "oclc_isa_version_803",
        "oclc_isa_version_900",  # isa version files (if the linker throws out
        "oclc_isa_version_906",  # the unneeded ones or we filter for the arch we need)
        "oclc_isa_version_1030",
        "oclc_unsafe_math_off",
        "oclc_unsafe_math_on",
        "oclc_wavefrontsize64_on",
        "oclc_abi_version_500",
    ]

    bitcode_files = []
    for n in bitcode_names:
        p = join(rocdl_dir, n + ".bc")  # rocm >= 3.9
        if not exists(p):  # rocm <= 3.8
            p = join(rocdl_dir, n + ".amdgcn.bc")
        if exists(p):
            bitcode_files.append(p)
        elif "isa_version" not in n and n not in {"irif"}:
            raise RuntimeError("could not find bitcode " + n)

    return tvm.runtime.convert(bitcode_files)


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


def have_matrixcore(compute_version=None):
    """Either MatrixCore support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    Returns
    -------
    have_matrixcore : bool
        True if MatrixCore support is provided, False otherwise
    """
    if compute_version is None:
        if tvm.rocm(0).exist:
            compute_version = tvm.rocm(0).compute_version
        else:
            raise RuntimeError("No ROCm runtime found")
    major, _ = parse_compute_version(compute_version)
    # matrix core first introduced in 8.0
    if major >= 8:
        return True

    return False


@tvm._ffi.register_func("tvm_callback_rocm_get_arch")
def get_rocm_arch(rocm_path="/opt/rocm"):
    """Utility function to get the AMD GPU architecture

    Parameters
    ----------
    rocm_path : str
        The path to rocm installation directory

    Returns
    -------
    gpu_arch : str
        The AMD GPU architecture
    """
    gpu_arch = "gfx900"
    # check if rocm is installed
    if not os.path.exists(rocm_path):
        print("ROCm not detected, using default gfx900")
        return gpu_arch
    try:
        # Execute rocminfo command
        rocminfo_output = subprocess.check_output([f"{rocm_path}/bin/rocminfo"]).decode("utf-8")

        # Use regex to match the "Name" field
        match = re.search(r"Name:\s+(gfx\d+[a-zA-Z]*)", rocminfo_output)
        if match:
            gpu_arch = match.group(1)
        return gpu_arch
    except subprocess.CalledProcessError:
        print(
            f"Unable to execute rocminfo command, \
                please ensure ROCm is installed and you have an AMD GPU on your system.\
                    using default {gpu_arch}."
        )
        return gpu_arch


def find_rocm_path():
    """Utility function to find ROCm path

    Returns
    -------
    path : str
        Path to ROCm root.
    """
    if "ROCM_PATH" in os.environ:
        return os.environ["ROCM_PATH"]
    cmd = ["which", "hipcc"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = out.decode("utf-8").strip()
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(out, "../.."))
    rocm_path = "/opt/rocm"
    if os.path.exists(os.path.join(rocm_path, "bin/hipcc")):
        return rocm_path
    raise RuntimeError("Cannot find ROCm path")
