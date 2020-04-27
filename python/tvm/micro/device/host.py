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
"""Compilation and config definitions for the host emulated device"""
import sys

from . import create_micro_lib_base, register_device, gen_mem_layout, MemConstraint

DEVICE_ID = "host"
TOOLCHAIN_PREFIX = ""
WORD_SIZE_BITS = 64 if sys.maxsize > 2**32 else 32

# we pretend we only have 320kb in the default case, so we can use `gen_mem_layout`
DEFAULT_AVAILABLE_MEM = 3200000
DEFAULT_SECTION_CONSTRAINTS = {
    "text": (20480, MemConstraint.ABSOLUTE_BYTES),
    "rodata": (20480, MemConstraint.ABSOLUTE_BYTES),
    "data": (768, MemConstraint.ABSOLUTE_BYTES),
    "bss": (4096, MemConstraint.ABSOLUTE_BYTES),
    "args": (4096, MemConstraint.ABSOLUTE_BYTES),
    "heap": (262144, MemConstraint.ABSOLUTE_BYTES),
    "workspace": (64000, MemConstraint.ABSOLUTE_BYTES),
    "stack": (80, MemConstraint.ABSOLUTE_BYTES),
}

def create_micro_lib(obj_path, src_path, lib_type, options=None, lib_src_paths=None):
    """Wrapper over `create_micro_lib_base` to add device-specific options

    Parameters
    ----------
    obj_path : str
        path to generated object file

    src_path : str
        path to source file

    lib_type : micro.LibType
        whether to compile a MicroTVM runtime or operator library

    options : Optional[List[str]]
        additional options to pass to GCC

    lib_src_paths : Optional[List[str]]
        paths to additional source files to be compiled into the library
    """
    if options is None:
        options = []
    else:
        options = list(options)
    # Cannot increase optimization level on host due to code loading method.
    options.append("-O0")
    if sys.maxsize > 2**32 and sys.platform.startswith("linux"):
        options += ["-mcmodel=large"]
    create_micro_lib_base(
        obj_path, src_path, TOOLCHAIN_PREFIX, DEVICE_ID, lib_type, options=options,
        lib_src_paths=lib_src_paths)


def generate_config(available_mem=None, section_constraints=None):
    """Generates a configuration for the host emulated device

    Parameters
    ----------
    available_mem: int
        number of RW bytes available for use on device

    section_constraints: Optional[Dict[str, Dict[Number, MemConstraint]]]
        maps section name to the quantity of available memory

    Return
    ------
    config : Dict[str, Any]
        MicroTVM config dict for this device
    """
    if available_mem is None:
        available_mem = DEFAULT_AVAILABLE_MEM
    if section_constraints is None:
        section_constraints = DEFAULT_SECTION_CONSTRAINTS
    mem_layout = gen_mem_layout(0, available_mem, WORD_SIZE_BITS, section_constraints)
    # TODO the host emulated device is an outlier, since we don't know how what
    # its base address will be until we've created it in the C++. is there any
    # way to change the infrastructure around this so it's not so much of an
    # outlier?

    # need to zero out all start addresses, because they don't make sense for a
    # host device (the memory region is allocated in the backend)
    for section in mem_layout:
        mem_layout[section]["start"] = 0
    return {
        "device_id": DEVICE_ID,
        "toolchain_prefix": TOOLCHAIN_PREFIX,
        "mem_layout": mem_layout,
        "word_size_bits": WORD_SIZE_BITS,
        "thumb_mode": False,
        "use_device_timer": False,
        "comms_method": "host",
    }


register_device(DEVICE_ID, {
    "create_micro_lib": create_micro_lib,
    "generate_config": generate_config,
})
