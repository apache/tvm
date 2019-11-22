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
"""Compilation and config definitions for Spike, a RISC-V functional ISA simulator"""
from collections import OrderedDict

from . import create_micro_lib_base, register_device

DEVICE_ID = "riscv_spike"
TOOLCHAIN_PREFIX = "riscv64-unknown-elf-"

def create_micro_lib(obj_path, src_path, lib_type, options=None):
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
    """
    create_micro_lib_base(
        obj_path,
        src_path,
        TOOLCHAIN_PREFIX,
        DEVICE_ID,
        lib_type,
        options=options)


def default_config(base_addr, server_addr, server_port):
    """Generates a default configuration for Spike

    Parameters
    ----------
    base_addr : int
        base address of the simulator (for calculating the memory layout)

    server_addr : str
        address of OpenOCD server to connect to

    server_port : int
        port of OpenOCD server to connect to

    Return
    ------
    config : Dict[str, Any]
        MicroTVM config dict for this device
    """
    res = {
        "device_id": DEVICE_ID,
        "toolchain_prefix": TOOLCHAIN_PREFIX,
        "mem_layout": OrderedDict([
            ("text", {
                "size": 20480,
            }),
            ("rodata", {
                "size": 20480,
            }),
            ("data", {
                "size": 768,
            }),
            ("bss", {
                "size": 768,
            }),
            ("args", {
                "size": 1280,
            }),
            ("heap", {
                "size": 262144,
            }),
            ("workspace", {
                "size": 20480,
            }),
            ("stack", {
                "size": 80,
            }),
        ]),
        "word_size": 4,
        "thumb_mode": True,
        "comms_method": "openocd",
        "server_addr": server_addr,
        "server_port": server_port,
    }
    # generate section start addresses from the given `base_addr`
    curr_offset = 0
    mem_layout = res["mem_layout"]
    for region_dict in mem_layout.values():
        region_dict["start"] = base_addr + curr_offset
        curr_offset += region_dict["size"]
    return res


register_device(DEVICE_ID, {
    "create_micro_lib": create_micro_lib,
    "default_config": default_config,
})
