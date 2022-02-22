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
"""Library information."""
from __future__ import absolute_import
import sys
import os

from .environment import get_vta_hw_path


def _get_lib_name(lib_name):
    """Get lib name with extension

    Returns
    -------
    lib_name_ext : str
        Name of VTA shared library with extension

    Parameters
    ------------
    lib_name : str
        Name of VTA shared library
    """
    if sys.platform.startswith("win32"):
        return lib_name + ".dll"
    if sys.platform.startswith("darwin"):
        return lib_name + ".dylib"
    return lib_name + ".so"


def find_libvta(lib_vta, optional=False):
    """Find VTA Chisel-based library

    Returns
    -------
    lib_found : str
        Library path

    Parameters
    ------------
    lib_vta : str
        Name of VTA shared library

    optional : bool
        Enable error check
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    tvm_library_path = os.environ.get("TVM_LIBRARY_PATH", None)
    if tvm_library_path is None:
        tvm_library_path = os.path.join(
            curr_path,
            os.pardir,
            os.pardir,
            os.pardir,
            "build",
        )

    lib_search = [tvm_library_path, os.path.join(get_vta_hw_path(), "build")]
    lib_name = _get_lib_name(lib_vta)
    lib_path = [os.path.join(x, lib_name) for x in lib_search]
    lib_found = [x for x in lib_path if os.path.exists(x)]
    if not lib_found and not optional:
        raise RuntimeError(
            "Cannot find the files.\n" + "List of candidates:\n" + str("\n".join(lib_path))
        )
    return lib_found
