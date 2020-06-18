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
"""Base definitions for MicroTVM config"""
import glob
import os
from pathlib import Path

from tvm.contrib import util as _util
from tvm.contrib.binutil import run_cmd
from tvm._ffi.libinfo import find_include_path
from tvm.micro import LibType, get_micro_host_driven_dir, get_micro_device_dir

_DEVICE_REGISTRY = {}

def register_device(device_id, device_funcs):
    """Register a device and associated compilation/config functions

    Parameters
    ----------
    device_id : str
        unique identifier for the device

    device_funcs : Dict[str, func]
        dictionary with compilation and config generation functions as values
    """
    if device_id in _DEVICE_REGISTRY:
        raise RuntimeError(f"\"{device_id}\" already exists in the device registry")
    _DEVICE_REGISTRY[device_id] = device_funcs


def get_device_funcs(device_id):
    """Get compilation and config generation functions for device

    Parameters
    ----------
    device_id : str
        unique identifier for the device

    Return
    ------
    device_funcs : Dict[str, func]
        dictionary with compilation and config generation functions as values
    """
    if device_id not in _DEVICE_REGISTRY:
        raise RuntimeError(f"\"{device_id}\" does not exist in the binutil registry")
    device_funcs = _DEVICE_REGISTRY[device_id]
    return device_funcs


def create_micro_lib_base(
        out_obj_path,
        in_src_path,
        toolchain_prefix,
        device_id,
        lib_type,
        options=None):
    """Compiles code into a binary for the target micro device.

    Parameters
    ----------
    out_obj_path : str
        path to generated object file

    in_src_path : str
        path to source file

    toolchain_prefix : str
        toolchain prefix to be used. For example, a prefix of
        "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
        the compiler and "riscv64-unknown-elf-ld" is used as the linker,
        etc.

    device_id : str
        unique identifier for the target device

    lib_type : micro.LibType
        whether to compile a MicroTVM runtime or operator library

    options : List[str]
        additional options to pass to GCC
    """
    base_compile_cmd = [
        f"{toolchain_prefix}gcc",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "--pedantic",
        "-c",
        "-O0",
        "-g",
        "-nostartfiles",
        "-nodefaultlibs",
        "-nostdlib",
        "-fdata-sections",
        "-ffunction-sections",
        ]
    if options is not None:
        base_compile_cmd += options

    src_paths = []
    include_paths = find_include_path() + [get_micro_host_driven_dir()]
    tmp_dir = _util.tempdir()
    # we might transform the src path in one of the branches below
    new_in_src_path = in_src_path
    if lib_type == LibType.RUNTIME:
        dev_dir = _get_device_source_dir(device_id)
        dev_src_paths = glob.glob(f"{dev_dir}/*.[csS]")
        # there needs to at least be a utvm_timer.c file
        assert dev_src_paths
        assert "utvm_timer.c" in map(os.path.basename, dev_src_paths)
        src_paths += dev_src_paths
    elif lib_type == LibType.OPERATOR:
        # create a temporary copy of the source, so we can inject the dev lib
        # header without modifying the original.
        temp_src_path = tmp_dir.relpath("temp.c")
        with open(in_src_path, "r") as f:
            src_lines = f.read().splitlines()
        src_lines.insert(0, "#include \"utvm_device_dylib_redirect.c\"")
        with open(temp_src_path, "w") as f:
            f.write("\n".join(src_lines))
        new_in_src_path = temp_src_path
        base_compile_cmd += ["-c"]
    else:
        raise RuntimeError("unknown lib type")

    src_paths += [new_in_src_path]

    for path in include_paths:
        base_compile_cmd += ["-I", path]

    prereq_obj_paths = []
    for src_path in src_paths:
        curr_obj_path = Path(src_path).with_suffix(".o").name
        assert curr_obj_path not in prereq_obj_paths
        prereq_obj_paths.append(curr_obj_path)
        curr_compile_cmd = base_compile_cmd + [src_path, "-o", curr_obj_path]
        run_cmd(curr_compile_cmd)

    ld_cmd = [f"{toolchain_prefix}ld", "-relocatable"]
    ld_cmd += prereq_obj_paths
    ld_cmd += ["-o", out_obj_path]
    run_cmd(ld_cmd)


def _get_device_source_dir(device_id):
    """Grabs the source directory for device-specific uTVM files"""
    dev_subdir = "/".join(device_id.split("."))
    return get_micro_device_dir() + "/" + dev_subdir
