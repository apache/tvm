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
"""Hexagon shared-library linking and custom linker registration."""

import os
from pathlib import Path

import tvm_ffi

from tvm.ir import StringImm
from tvm.support import cc

HEXAGON_TOOLCHAIN = os.environ.get("HEXAGON_TOOLCHAIN", "")
HEXAGON_LINK_MAIN = Path(HEXAGON_TOOLCHAIN) / "bin" / "hexagon-link"


def register_linker(f):
    """Register a function that will return the path to the Hexagon linker."""
    return tvm_ffi.register_global_func("tvm.contrib.hexagon.hexagon_link", f, True)


@tvm_ffi.register_global_func("tvm.contrib.hexagon.hexagon_link")
def hexagon_link() -> str:
    """Return path to the Hexagon linker."""
    return str(HEXAGON_LINK_MAIN)


@tvm_ffi.register_global_func("tvm.contrib.hexagon.link_shared")
def link_shared(so_name, objs, extra_args=None):
    """Link shared library on Hexagon using the registered Hexagon linker.

    Parameters
    ----------
    so_name : str
        Name of the shared library file.
    objs : list[str, tvm.ir.StringImm]
    extra_args : dict (str->str) or Map<String,String>
        Additional arguments:
            'hex_arch' - Hexagon architecture, e.g. v68
            'verbose'  - Print additional information if the key is present

    Returns
    -------
    ret_val : int
        This function returns 0 at the moment.
    """

    # The list of object files can be passed as built-in Python strings,
    # or as tvm.ir.StringImm's.
    def to_str(s):
        if isinstance(s, StringImm):
            return s.value
        assert isinstance(s, str), 'argument "' + str(s) + '" should be a string or StrImm'
        return s

    objs = [to_str(s) for s in objs]

    if not extra_args:
        extra_args = {}
    hex_arch = extra_args.get("hex_arch") or "v68"
    linker = tvm_ffi.get_global_func("tvm.contrib.hexagon.hexagon_link")()
    if extra_args.get("verbose"):
        print("tvm.contrib.hexagon.link_shared:")
        print("  Using linker:", linker)
        print("  Library name:", so_name)
        print("  Object files:", objs)
        print("  Architecture:", hex_arch)
    if not os.access(linker, os.X_OK):
        message = 'The linker "' + linker + '" does not exist or is not executable.'
        if not os.environ.get("HEXAGON_TOOLCHAIN"):
            message += (
                " The environment variable HEXAGON_TOOLCHAIN is unset. Please export "
                + "HEXAGON_TOOLCHAIN in your environment, so that ${HEXAGON_TOOLCHAIN}/bin/"
                + "hexagon-link exists."
            )
        else:
            message += (
                " Please verify the value of the HEXAGON_LINKER environment variable "
                + '(currently set to "'
                + HEXAGON_TOOLCHAIN
                + '").'
            )
        raise Exception(message)

    libpath = os.path.join(HEXAGON_TOOLCHAIN, "target", "hexagon", "lib", hex_arch, "G0")
    cc.create_shared(
        so_name,
        objs,
        # pylint: disable=bad-whitespace
        options=[
            "-Bdynamic",
            "-shared",
            "-export-dynamic",
            os.path.join(libpath, "pic", "libgcc.so"),
        ],
        cc=linker,
    )
    return 0
