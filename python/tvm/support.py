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
"""Support infra of TVM."""

import json
import os
import sys
import textwrap

import tvm_ffi

import tvm

from . import get_global_func

tvm_ffi.init_ffi_api("support", __name__)


def detect_active_modules() -> dict:
    """Detect device-runtime modules linked into the current libtvm
    by querying the FFI global function registry for
    ``ffi.Module.create.<kind>`` registrations.

    Probes a minimal set of key device runtimes (cuda, vulkan, opencl);
    expand the list when a new caller needs it.

    Returns
    -------
    active : dict[str, bool]
        Mapping from runtime kind to whether it is registered in this build.
    """
    # Registry: "ffi.Module.create.<kind>" — per-backend device-module factory.
    # Grep hint: grep -rn 'ffi.Module.create.' src/ python/
    keys = ["cuda", "vulkan", "opencl"]
    return {
        k: get_global_func(f"ffi.Module.create.{k}", allow_missing=True) is not None for k in keys
    }


def describe():
    """
    Print out information about TVM and the current Python environment
    """
    print("Python Environment")
    sys_version = sys.version.replace("\n", " ")
    uname = os.uname()
    uname = f"{uname.sysname} {uname.release} {uname.version} {uname.machine}"
    lines = [
        f"TVM version    = {tvm.__version__}",
        f"Python version = {sys_version} ({sys.maxsize.bit_length() + 1} bit)",
        f"os.uname()     = {uname}",
    ]
    print(textwrap.indent("\n".join(lines), prefix="  "))
    print("Active Device Runtimes:")
    print(textwrap.indent(json.dumps(detect_active_modules(), indent=2), prefix="  "))
