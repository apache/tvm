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

from . import create_micro_lib_base, register_device

DEVICE_ID = "host"
TOOLCHAIN_PREFIX = ""

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
    if options is None:
        options = []
    if sys.maxsize > 2**32 and sys.platform.startswith("linux"):
        options += ["-mcmodel=large"]
    create_micro_lib_base(
        obj_path, src_path, TOOLCHAIN_PREFIX, DEVICE_ID, lib_type, options=options)


def default_config():
    """Generates a default configuration for the host emulated device

    Return
    ------
    config : Dict[str, Any]
        MicroTVM config dict for this device
    """
    return {
        "device_id": DEVICE_ID,
        "toolchain_prefix": TOOLCHAIN_PREFIX,
        "mem_layout": {
            "text": {
                "size": 20480,
            },
            "rodata": {
                "size": 20480,
            },
            "data": {
                "size": 768,
            },
            "bss": {
                "size": 768,
            },
            "args": {
                "size": 1280,
            },
            "heap": {
                "size": 262144,
            },
            "workspace": {
                "size": 20480,
            },
            "stack": {
                "size": 80,
            },
        },
        "word_size": 8 if sys.maxsize > 2**32 else 4,
        "thumb_mode": False,
        "comms_method": "host",
    }


register_device(DEVICE_ID, {
    "create_micro_lib": create_micro_lib,
    "default_config": default_config,
})
