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
import subprocess
from os.path import join, exists
from . import util
from .._ffi.base import py_str
from ..api import register_func, convert

def rocm_link(in_file, out_file):
    """Link relocatable ELF object to shared ELF object using lld

    Parameters
    ----------
    in_file : str
        Input file name (relocatable ELF object file)

    out_file : str
        Output file name (shared ELF object file)
    """
    args = ["ld.lld", "-shared", in_file, "-o", out_file]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Linking error using ld.lld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


@register_func("tvm_callback_rocm_link")
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
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("rocm_kernel.o")
    tmp_cobj = tmp_dir.relpath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    rocm_link(tmp_obj, tmp_cobj)
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin

@register_func("tvm_callback_rocm_bitcode_path")
def callback_rocm_bitcode_path(rocdl_dir="/opt/rocm/lib/"):
    """Utility function to find ROCm device library bitcodes

    Parameters
    ----------
    rocdl_dir : str
        The path to rocm library directory
        The default value is the standard location
    """
    # seems link order matters.
    bitcode_files = [
        "oclc_daz_opt_on.amdgcn.bc",
        "ocml.amdgcn.bc",
        "hc.amdgcn.bc",
        "irif.amdgcn.bc",
        "ockl.amdgcn.bc",
        "oclc_correctly_rounded_sqrt_off.amdgcn.bc",
        "oclc_correctly_rounded_sqrt_on.amdgcn.bc",
        "oclc_daz_opt_off.amdgcn.bc",
        "oclc_finite_only_off.amdgcn.bc",
        "oclc_finite_only_on.amdgcn.bc",
        "oclc_isa_version_803.amdgcn.bc",
        "oclc_isa_version_900.amdgcn.bc",
        "oclc_unsafe_math_off.amdgcn.bc",
        "oclc_unsafe_math_on.amdgcn.bc"
    ]
    paths = [join(rocdl_dir, bitcode) for bitcode in bitcode_files]
    return convert([path for path in paths if exists(path)])
