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
import textwrap
import ctypes
import os
import re
import sys

import tvm
import tvm._ffi
from .runtime.module import Module
from . import get_global_func

tvm._ffi._init_api("support", __name__)


def libinfo():
    """Returns a dictionary containing compile-time info, including cmake flags and git commit hash

    Returns
    -------
    info: Dict[str, str]
        The dictionary of compile-time info.
    """
    get_lib_info_func = get_global_func("support.GetLibInfo", allow_missing=True)
    if get_lib_info_func is not None:
        lib_info = get_lib_info_func()
        if lib_info is None:
            return {}
    else:
        return {}
    return dict(lib_info.items())


def describe():
    """
    Print out information about TVM and the current Python environment
    """
    info = list((k, v) for k, v in libinfo().items())
    info = dict(sorted(info, key=lambda x: x[0]))
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
    print("CMake Options:")
    print(textwrap.indent(json.dumps(info, indent=2), prefix="  "))


class FrontendTestModule(Module):
    """A tvm.runtime.Module whose member functions are PackedFunc."""

    def __init__(self, entry_name=None):
        underlying_mod = get_global_func("testing.FrontendTestModule")()
        handle = underlying_mod.handle

        # Set handle to NULL to avoid cleanup in c++ runtime, transferring ownership.
        # Both cython and ctypes FFI use c_void_p, so this is safe to assign here.
        underlying_mod.handle = ctypes.c_void_p(0)

        super(FrontendTestModule, self).__init__(handle)
        if entry_name is not None:
            self.entry_name = entry_name

    def add_function(self, name, func):
        self.get_function("__add_function")(name, func)

    def __setitem__(self, key, value):
        self.add_function(key, value)


@tvm._ffi.register_func("tvm.support.regex_match")
def _regex_match(regex_pattern: str, match_against: str) -> bool:
    """Check if a pattern matches a regular expression

    This function should be used instead of `std::regex` within C++
    call sites, to avoid ABI incompatibilities with pytorch.

    Currently, the pytorch wheels available through pip install use
    the pre-C++11 ABI by setting `-DUSE_CXX11_ABI=0` [0]. If TVM were to
    user the pre-C++11 ABI, this would cause breakages with
    dynamically-linked LLVM environments.

    Use of the `<regex>` header in TVM should be avoided, as its
    implementation is not supported by gcc's dual ABI. This ABI
    incompatibility results in runtime errors either when `std::regex`
    is called from TVM, or when `std::regex` is called from pytorch,
    depending on which library was loaded first.  This restriction can
    be removed when a version of pytorch compiled using
    `-DUSE_CXX11_ABI=1` is available from PyPI.

    [0] https://github.com/pytorch/pytorch/issues/51039

    Parameters
    ----------
    regex_pattern: str

         The regular expression

    match_against: str

        The string against which to match the regular expression

    Returns
    -------
    match_result: bool

        True if `match_against` matches the pattern defined by
        `regex_pattern`, and False otherwise.
    """
    match = re.match(regex_pattern, match_against)
    return match is not None
